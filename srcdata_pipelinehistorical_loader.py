"""
Historical Data Loader - Phase 1 Foundation
Responsible for creating and validating data files before any consumption.
Implements multi-source data collection with credibility validation.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np
import ccxt
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for data source reliability"""
    name: str
    priority: int
    max_retries: int
    timeout_seconds: int = 30
    rate_limit_requests: int = 10
    rate_limit_seconds: int = 1

class HistoricalDataLoader:
    """Creates and manages historical market data with integrity validation"""
    
    # Approved exchange sources in priority order
    EXCHANGE_SOURCES = [
        ('binance', 1),
        ('kraken', 2),
        ('coinbasepro', 3)
    ]
    
    # Initial token pairs for analysis
    TOKEN_PAIRS = [
        'BTC/USDT',
        'ETH/USDT',
        'SOL/USDT',
        'AVAX/USDT'
    ]
    
    def __init__(self, data_dir: str = 'data/csv'):
        """Initialize loader with directory structure verification"""
        self.data_dir = Path(data_dir)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.data_credibility_threshold = 0.95  # 95% consensus required
        
        # Create directory structure if it doesn't exist
        self._initialize_directories()
        
    def _initialize_directories(self) -> None:
        """Create all necessary directories before file operations"""
        directories = [
            self.data_dir,
            self.data_dir / 'raw',
            self.data_dir / 'processed',
            self.data_dir / 'validation'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Verified/Created directory: {directory}")
    
    def initialize_exchanges(self) -> bool:
        """Initialize exchange connections with error handling"""
        successful_init = 0
        
        for exchange_name, priority in self.EXCHANGE_SOURCES:
            try:
                # Get exchange class from ccxt
                exchange_class = getattr(ccxt, exchange_name)
                
                # Initialize with conservative rate limiting
                exchange = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,
                    'verbose': False
                })
                
                # Test connection with minimal request
                exchange.load_markets()
                self.exchanges[exchange_name] = exchange
                successful_init += 1
                
                logger.info(f"Successfully initialized {exchange_name} (Priority {priority})")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {exchange_name}: {str(e)[:100]}")
                continue
        
        # Require at least 2 sources for data credibility
        if successful_init >= 2:
            logger.info(f"Successfully initialized {successful_init} data sources")
            return True
        else:
            logger.error(f"Insufficient data sources: {successful_init}/3 initialized")
            return False
    
    def fetch_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: str = '1m',
        days_back: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with multi-source validation
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (1m, 5m, 1h, etc.)
            days_back: Number of days of historical data
        
        Returns:
            DataFrame with OHLCV data or None if insufficient credible data
        """
        all_data: List[pd.DataFrame] = []
        start_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        logger.info(f"Fetching {symbol} {timeframe} data for {days_back} days")
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Check if symbol exists on exchange
                if symbol not in exchange.symbols:
                    logger.debug(f"{symbol} not available on {exchange_name}")
                    continue
                
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=start_timestamp,
                    limit=1000  # Multiple calls will be made if needed
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add source identifier
                df['source'] = exchange_name
                
                all_data.append(df)
                logger.info(f"Fetched {len(df)} candles from {exchange_name}")
                
            except ccxt.RateLimitExceeded:
                logger.warning(f"Rate limit exceeded on {exchange_name}, waiting...")
                asyncio.sleep(1)
                continue
            except Exception as e:
                logger.error(f"Error fetching from {exchange_name}: {str(e)[:100]}")
                continue
        
        if len(all_data) < 2:
            logger.error(f"Insufficient data sources for {symbol}: {len(all_data)} sources")
            return None
        
        # Validate data credibility across sources
        validated_data = self._validate_data_consistency(all_data, symbol)
        
        if validated_data is not None:
            # Save to CSV - CREATING FILE BEFORE READING
            safe_symbol = symbol.replace('/', '_')
            filename = f"{safe_symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = self.data_dir / 'raw' / filename
            
            validated_data.to_csv(filepath)
            logger.info(f"Created data file: {filepath} with {len(validated_data)} rows")
            
            # Also create processed version
            processed_data = self._create_processed_dataset(validated_data)
            processed_path = self.data_dir / 'processed' / filename
            processed_data.to_csv(processed_path)
            
            return processed_data
        
        return None
    
    def _validate_data_consistency(
        self, 
        dataframes: List[pd.DataFrame], 
        symbol: str
    ) -> Optional[pd.DataFrame]:
        """
        Validate data consistency across multiple sources
        Returns consensus data where sources agree within threshold
        """
        if len(dataframes) < 2:
            return None
        
        # Align timestamps
        aligned_data = []
        min_length = min(len(df) for df in dataframes)
        
        for df in dataframes:
            # Take most recent common data
            aligned_df = df.iloc[-min_length:].copy()
            aligned_data.append(aligned_df)
        
        # Calculate price consensus
        consensus_records = []
        
        for i in range(min_length):
            timestamps = [df.index[i] for df in aligned_data]
            
            # Check timestamp alignment (within 1 minute)
            max_time_diff = max(timestamps) - min(timestamps)
            if max_time_diff > timedelta(minutes=1):
                logger.warning(f"Timestamp misalignment at index {i}: {max_time_diff}")
                continue
            
            # Get closing prices from all sources
            closes = [df.iloc[i]['close'] for df in aligned_data]
            
            # Calculate median and standard deviation
            median_close = np.median(closes)
            std_close = np.std(closes)
            
            # Check if values are within 2% of median
            valid_prices = [
                price for price in closes 
                if abs(price - median_close) / median_close < 0.02
            ]
            
            # Require at least 2/3 sources in consensus
            consensus_ratio = len(valid_prices) / len(closes)
            
            if consensus_ratio >= self.data_credibility_threshold:
                # Use median of valid prices
                consensus_close = np.median(valid_prices) if valid_prices else median_close
                
                # Build consensus record
                record = {
                    'timestamp': timestamps[0],
                    'open': np.median([df.iloc[i]['open'] for df in aligned_data]),
                    'high': np.median([df.iloc[i]['high'] for df in aligned_data]),
                    'low': np.median([df.iloc[i]['low'] for df in aligned_data]),
                    'close': consensus_close,
                    'volume': np.sum([df.iloc[i]['volume'] for df in aligned_data]),
                    'sources_used': len(valid_prices),
                    'price_variance': float(std_close / median_close)
                }
                consensus_records.append(record)
            else:
                logger.warning(f"Low consensus at {timestamps[0]}: {consensus_ratio:.1%}")
        
        if consensus_records:
            df_consensus = pd.DataFrame(consensus_records)
            df_consensus.set_index('timestamp', inplace=True)
            
            logger.info(f"Created consensus data for {symbol}: {len(df_consensus)}/{min_length} rows validated")
            return df_consensus
        
        logger.error(f"No consensus data achieved for {symbol}")
        return None
    
    def _create_processed_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features for volatility analysis"""
        processed_df = df.copy()
        
        # Calculate returns
        processed_df['returns'] = processed_df['close'].pct_change()
        
        # Calculate volatility metrics (rolling windows)
        for window in [5, 15, 30, 60]:
            processed_df[f'volatility_{window}'] = (
                processed_df['returns'].rolling(window=window).std() * np.sqrt(24 * 60)  # Annualized
            )
        
        # Calculate spread metrics
        processed_df['spread_pct'] = (
            (processed_df['high'] - processed_df['low']) / processed_df['close']
        )
        
        # Volume profile
        processed_df['volume_ma'] = processed_df['volume'].rolling(window=30).mean()
        processed_df['volume_ratio'] = processed_df['volume'] / processed_df['volume_ma']
        
        # Remove NaN values from rolling calculations
        processed_df = processed_df.dropna()
        
        return processed_df
    
    def load_all_pairs(self) -> Dict[str, pd.DataFrame]:
        """Load data for all configured token pairs"""
        if not self.initialize_exchanges():
            logger.error("Failed to initialize exchanges. Cannot load data.")
            return {}
        
        all_data = {}
        
        for pair in self.TOKEN_PAIRS:
            logger.info(f"Processing pair: {pair}")
            
            data = self.fetch_ohlcv_data(pair, '1m', 30)
            
            if data is not None:
                all_data[pair] = data
                
                # Log summary statistics
                logger.info(f"  Rows: {len(data)}")
                logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
                logger.info(f"  Mean volatility: {data['volatility_30'].mean():.4f}")
            else:
                logger.error(f"  Failed to load data for {pair}")
        
        return all_data

def main():
    """Main execution function"""
    logger.info("=== Starting Historical Data Loader ===")
    
    loader = HistoricalDataLoader()
    all_data = loader.load_all_pairs()
    
    if all_data:
        logger.info(f"Successfully loaded {len(all_data)} pairs")
        
        # Log file creation summary
        data_dir = Path('data/csv')
        raw_files = list((data_dir / 'raw').glob('*.csv'))
        processed_files = list((data_dir / 'processed').glob('*.csv'))
        
        logger.info(f"Created {len(raw_files)} raw data files")
        logger.info(f"Created {len(processed_files)} processed data files")
        
        return True
    else:
        logger.error("No data loaded successfully")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)