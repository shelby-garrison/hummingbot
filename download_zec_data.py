#!/usr/bin/env python3
"""
Simple ZEC/USD Data Downloader
Downloads ZEC/USD candle data without requiring the full task system
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the data source
try:
    from core.data_sources import CLOBDataSource
    CLOB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CLOBDataSource not available: {e}")
    CLOB_AVAILABLE = False


class SimpleZECDataDownloader:
    """Simple data downloader for ZEC/USD without full task system"""
    
    def __init__(self):
        self.clob = None
        if CLOB_AVAILABLE:
            self.clob = CLOBDataSource()
    
    async def download_zec_data(self, days_back: int = 30):
        """Download ZEC/USD data for multiple timeframes"""
        print("ZEC/USD Data Downloader")
        print("=" * 50)
        
        if not CLOB_AVAILABLE:
            print("[ERROR] CLOBDataSource not available. Please install required dependencies:")
            print("   pip install hummingbot")
            return False
        
        # Configuration
        connector_name = "binance_perpetual"
        trading_pair = "ZEC-USDT"
        intervals = ["5m", "30m", "1h", "1d"]
        
        print(f"Downloading ZEC/USD data for {days_back} days")
        print(f"Timeframes: {', '.join(intervals)}")
        print(f"Exchange: {connector_name}")
        print()
        
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)
        
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        print(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Track statistics
        stats = {
            "intervals_processed": 0,
            "candles_downloaded": 0,
            "errors": 0
        }
        
        # Download data for each timeframe
        for interval in intervals:
            try:
                print(f"Downloading {trading_pair} {interval} data...")
                
                # Download candles
                candles = await self.clob.get_candles(
                    connector_name,
                    trading_pair,
                    interval,
                    start_timestamp,
                    end_timestamp
                )
                
                if candles.data.empty:
                    print(f"  [WARNING] No data available for {interval}")
                    continue
                
                # Save to parquet file
                filename = f"{connector_name}|{trading_pair}|{interval}.parquet"
                filepath = f"app/data/cache/candles/{filename}"
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save data
                candles.data.to_parquet(filepath)
                
                stats["candles_downloaded"] += len(candles.data)
                stats["intervals_processed"] += 1
                
                print(f"  [OK] Downloaded {len(candles.data)} candles")
                print(f"  [FILE] Saved to: {filepath}")
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                stats["errors"] += 1
                print(f"  [ERROR] Error downloading {interval}: {e}")
                continue
        
        # Save all cached data
        if self.clob:
            print("\nSaving cached data...")
            self.clob.dump_candles_cache()
        
        # Print summary
        print("\n" + "=" * 50)
        print("DOWNLOAD SUMMARY")
        print("=" * 50)
        print(f"Intervals processed: {stats['intervals_processed']}/{len(intervals)}")
        print(f"Total candles downloaded: {stats['candles_downloaded']}")
        print(f"Errors: {stats['errors']}")
        
        if stats["intervals_processed"] > 0:
            print("\n[OK] Data download completed successfully!")
            print("\nNext steps:")
            print("1. Run: python simple_zec_backtest.py")
            print("2. Or run: python zec_backtest_runner.py (requires full setup)")
            return True
        else:
            print("\n[ERROR] No data was downloaded")
            return False


async def main():
    """Main execution function"""
    downloader = SimpleZECDataDownloader()
    
    try:
        success = await downloader.download_zec_data(days_back=30)
        
        if success:
            print("\n[SUCCESS] Ready for backtesting!")
        else:
            print("\n[INFO] Try installing dependencies:")
            print("   pip install hummingbot")
            
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
