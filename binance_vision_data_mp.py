"""
币安历史数据多进程下载工具 (data.binance.vision版本)
从 https://data.binance.vision 下载公开的历史K线数据
特点：
1. 无频率限制（公开数据）
2. 数据格式为ZIP压缩的CSV文件
3. 按天存储，每个文件对应一天的数据
4. 支持现货(spot)和合约(futures/um)数据
"""

import os
import time
import zipfile
import io
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional
from pathlib import Path

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from tenacity import retry, stop_after_attempt, wait_fixed


# ==================== 工具函数 ====================

def get_date_list(start_str: str, end_str: str) -> List[str]:
    """
    获取日期列表
    
    Args:
        start_str: 开始日期 "YYYY-MM-DD"
        end_str: 结束日期 "YYYY-MM-DD"
        
    Returns:
        日期列表 ["YYYY-MM-DD", ...]
    """
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d")
    
    date_list = []
    current_dt = start_dt
    
    while current_dt <= end_dt:
        date_list.append(current_dt.strftime("%Y-%m-%d"))
        current_dt += timedelta(days=1)
    
    return date_list


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_with_retry(url: str, timeout: int = 30) -> bytes:
    """带重试机制的HTTP下载"""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def parse_csv_to_dataframe(csv_content: bytes, symbol: str, date: str) -> pd.DataFrame:
    """
    解析CSV内容为DataFrame
    
    CSV格式：
    - 旧格式（无表头）：时间戳为13位毫秒
    - 新格式（有表头）：时间戳可能是13位毫秒或16位微秒
    
    字段：
    0: open_time - 开盘时间
    1: open - 开盘价
    2: high - 最高价
    3: low - 最低价
    4: close - 收盘价
    5: volume - 交易量
    6: close_time - 收盘时间
    7: quote_asset_volume - 成交额
    8: number_of_trades - 交易次数
    9: taker_buy_base_asset_volume - 主动买入量
    10: taker_buy_quote_asset_volume - 主动买入额
    11: ignore - 忽略
    """
    try:
        # 先尝试读取CSV判断是否有表头
        first_line = csv_content.decode('utf-8').split('\n')[0]
        has_header = 'open_time' in first_line or 'open' in first_line[:50]
        
        column_names = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trade_cnt',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ]
        
        if has_header:
            # 有表头，直接读取
            df = pd.read_csv(io.BytesIO(csv_content))
            # 重命名列名以匹配我们的标准（如果需要）
            df.columns = column_names
        else:
            # 无表头，指定列名
            df = pd.read_csv(io.BytesIO(csv_content), names=column_names)
        
        # 选择和重命名需要的列
        df = df.rename(columns={
            'open_time': 'timestamp',
            'quote_volume': 'value',
            'taker_buy_volume': 'active_buy_volume',
            'taker_buy_quote_volume': 'active_buy_value'
        })
        
        # 选择需要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                 'value', 'trade_cnt', 'active_buy_volume', 'active_buy_value']]
        
        # 转换时间戳 - 使用 datetime 替代 time.localtime 以支持未来日期（如2025年）
        # 支持毫秒（13位）和微秒（16位）时间戳
        def convert_timestamp(x):
            ts = int(x)
            # 判断是毫秒还是微秒
            if ts > 1e15:  # 16位数字，微秒级别
                ts = ts / 1000000
            else:  # 13位数字，毫秒级别
                ts = ts / 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        df['timestamp'] = df['timestamp'].apply(convert_timestamp)
        
        # 重构数据格式：按日期和分钟展开
        df['mkt_date'] = df['timestamp'].str[:10]
        df['minute'] = df['timestamp'].str[-8:-3].str.replace(':', '')
        del df['timestamp']
        
        # 透视表：将分钟作为列
        df.set_index(['mkt_date', 'minute'], inplace=True)
        df = df.unstack(level='minute')
        df = df.swaplevel(0, 1, axis=1)
        df = df.sort_index(axis=1)
        df.columns = [f'{col[1]}_{col[0]}' for col in df.columns]
        
        # 添加交易对标识
        df = df.reset_index()
        df['instrument'] = symbol
        df.set_index(['mkt_date', 'instrument'], inplace=True)
        
        return df
        
    except Exception as e:
        print(f"解析CSV失败 {symbol} {date}: {str(e)}")
        return None


# ==================== 多进程工作函数 ====================

def download_single_task(task: dict) -> Tuple[str, str, str, bool]:
    """
    下载单个任务
    
    Args:
        task: 包含所有必要参数的字典
        
    Returns:
        (symbol, date, market_type, success): 任务执行结果
    """
    symbol = task['symbol']
    date = task['date']
    interval = task['interval']
    market_type = task['market_type']
    base_url = task['base_url']
    overwrite = task.get('overwrite', False)
    
    # 构建存储路径
    data_dir = f'data/daily/{market_type}/{interval}/{date}/'
    data_path = os.path.join(data_dir, f'{symbol}.pkl')
    
    # 根据overwrite参数决定是否跳过已存在的数据
    if os.path.exists(data_path) and not overwrite:
        return (symbol, date, market_type, True)
    
    try:
        # 构建下载URL
        # 格式：https://data.binance.vision/data/{market_path}/daily/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{date}.zip
        if market_type == 'spot':
            market_path = 'spot'
        else:  # futures
            market_path = 'futures/um'
        
        filename = f"{symbol}-{interval}-{date}.zip"
        url = f"{base_url}/data/{market_path}/daily/klines/{symbol}/{interval}/{filename}"
        
        # 下载ZIP文件
        zip_content = download_with_retry(url)
        
        # 解压并读取CSV
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
            # ZIP中应该只有一个CSV文件
            csv_filename = f"{symbol}-{interval}-{date}.csv"
            csv_content = zf.read(csv_filename)
        
        # 解析CSV为DataFrame
        df = parse_csv_to_dataframe(csv_content, symbol, date)
        
        if df is None or df.empty:
            # 空数据也保存一个None，避免重复下载
            os.makedirs(data_dir, exist_ok=True)
            pd.to_pickle(None, data_path)
            return (symbol, date, market_type, True)
        
        # 保存为pickle文件
        os.makedirs(data_dir, exist_ok=True)
        df.to_pickle(data_path)
        
        return (symbol, date, market_type, True)
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # 文件不存在（可能是交易对在该日期还未上线）
            # 保存None避免重复下载
            os.makedirs(data_dir, exist_ok=True)
            pd.to_pickle(None, data_path)
            return (symbol, date, market_type, True)
        else:
            print(f"\n下载失败 {symbol} {date} ({market_type}): HTTP {e.response.status_code}")
            return (symbol, date, market_type, False)
            
    except Exception as e:
        print(f"\n下载失败 {symbol} {date} ({market_type}): {str(e)}")
        return (symbol, date, market_type, False)


# ==================== 主类 ====================

class BinanceVisionData:
    """币安历史数据下载器（从 data.binance.vision）"""
    
    BASE_URL = "https://data.binance.vision"
    
    SUPPORT_INTERVAL = {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M"
    }
    
    SUPPORT_MARKET_TYPE = {"spot", "futures"}
    
    def __init__(self,
                 symbol_lst: List[str],
                 market_type: str = "spot",
                 interval: str = "1m",
                 start: str = "2017-08-17",
                 end: str = "2025-10-20",
                 overwrite: bool = False,
                 num_workers: Optional[int] = None):
        """
        初始化数据下载器
        
        Args:
            symbol_lst: 交易对列表，如 ['BTCUSDT', 'ETHUSDT']
            market_type: 市场类型 'spot'(现货) 或 'futures'(合约)
            interval: K线时间间隔
            start: 开始日期 "YYYY-MM-DD"
            end: 结束日期 "YYYY-MM-DD"
            overwrite: 是否覆盖已存在的数据，默认False
            num_workers: 工作进程数，默认为CPU核心数
        """
        self.symbol_lst = symbol_lst
        self.market_type = market_type
        self.interval = interval
        self.start = start
        self.end = end
        self.overwrite = overwrite
        self.num_workers = num_workers or mp.cpu_count()
        
        # 验证参数
        if self.market_type not in self.SUPPORT_MARKET_TYPE:
            raise ValueError(f"不支持的市场类型: {self.market_type}，支持的类型: {self.SUPPORT_MARKET_TYPE}")
        
        if self.interval not in self.SUPPORT_INTERVAL:
            raise ValueError(f"不支持的时间间隔: {self.interval}，支持的间隔: {self.SUPPORT_INTERVAL}")
    
    def download_klines(self):
        """下载K线数据（多进程版本）"""
        print(f"\n{'='*60}")
        print(f"开始下载K线数据 (data.binance.vision)")
        print(f"{'='*60}")
        print(f"数据源: {self.BASE_URL}")
        print(f"市场类型: {self.market_type} ({'现货' if self.market_type == 'spot' else '合约'})")
        print(f"交易对: {', '.join(self.symbol_lst)}")
        print(f"时间范围: {self.start} 至 {self.end}")
        print(f"时间间隔: {self.interval}")
        print(f"覆盖模式: {'是' if self.overwrite else '否（跳过已存在）'}")
        
        # 获取日期列表
        date_list = get_date_list(self.start, self.end)
        print(f"总天数: {len(date_list)} 天")
        print(f"总任务数: {len(self.symbol_lst)} 个交易对 × {len(date_list)} 天 = {len(self.symbol_lst) * len(date_list)} 个任务")
        print(f"进程数: {self.num_workers}")
        
        # 构建所有任务
        tasks = []
        for symbol in self.symbol_lst:
            for date in date_list:
                task = {
                    'symbol': symbol,
                    'date': date,
                    'interval': self.interval,
                    'market_type': self.market_type,
                    'base_url': self.BASE_URL,
                    'overwrite': self.overwrite
                }
                tasks.append(task)
        
        print(f"\n开始下载...\n")
        
        # 使用多进程下载
        success_count = 0
        fail_count = 0

        sub_tasks = [i for i in tasks if i['date'] == '2025-02-01']
        download_single_task(sub_tasks[0])

        with mp.Pool(self.num_workers) as pool:
            # 使用 imap_unordered 实时显示进度
            results = pool.imap_unordered(download_single_task, tasks)
            
            # 用 tqdm 包裹结果迭代器
            for symbol, date, market_type, success in tqdm(results,
                                                           total=len(tasks),
                                                           desc="下载进度",
                                                           unit="任务",
                                                           ncols=100):
                if success:
                    success_count += 1
                else:
                    fail_count += 1
        
        # 最终统计
        print(f"\n{'='*60}")
        print(f"下载完成！")
        print(f"{'='*60}")
        print(f"[成功] {success_count} 个任务")
        if fail_count > 0:
            print(f"[失败] {fail_count} 个任务")
        print(f"[总计] {success_count + fail_count} 个任务")
        print(f"{'='*60}\n")
        
        return success_count, fail_count


# ==================== 兼容类 ====================

class SpotVisionData(BinanceVisionData):
    """现货历史数据下载器"""
    
    def __init__(self,
                 symbol_lst: List[str],
                 interval: str = "1m",
                 start: str = "2017-08-17",
                 end: str = "2025-10-20",
                 overwrite: bool = False,
                 num_workers: Optional[int] = None):
        super().__init__(
            symbol_lst=symbol_lst,
            market_type='spot',
            interval=interval,
            start=start,
            end=end,
            overwrite=overwrite,
            num_workers=num_workers
        )


class FuturesVisionData(BinanceVisionData):
    """合约历史数据下载器"""
    
    def __init__(self,
                 symbol_lst: List[str],
                 interval: str = "1m",
                 start: str = "2017-08-17",
                 end: str = "2025-10-20",
                 overwrite: bool = False,
                 num_workers: Optional[int] = None):
        super().__init__(
            symbol_lst=symbol_lst,
            market_type='futures',
            interval=interval,
            start=start,
            end=end,
            overwrite=overwrite,
            num_workers=num_workers
        )


# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description='币安历史K线数据下载工具 (data.binance.vision)')
    parser.add_argument('--market', type=str, default='spot', choices=['spot', 'futures'],
                        help='市场类型: spot(现货) 或 futures(合约)')
    parser.add_argument('--symbols', type=str, default='BTCUSDT,ETHUSDT',
                        help='交易对列表，逗号分隔')
    parser.add_argument('--interval', type=str, default='1m',
                        help='K线时间间隔')
    parser.add_argument('--start', type=str, default='2017-08-17',
                        help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, default='9999-99-99',
                        help='结束日期 YYYY-MM-DD')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的数据')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='工作进程数，默认为CPU核心数')
    
    args = parser.parse_args()
    
    # 解析交易对列表
    symbol_lst = [s.strip() for s in args.symbols.split(',')]
    
    # # 创建下载器
    # downloader = BinanceVisionData(
    #     symbol_lst=symbol_lst,
    #     market_type=args.market,
    #     interval=args.interval,
    #     start=args.start,
    #     end=args.end,
    #     overwrite=args.overwrite,
    #     num_workers=args.num_workers
    # )    
    
    # 创建下载器
    downloader = BinanceVisionData(
        symbol_lst=['BTCUSDT' ,'ETHUSDT'],
        market_type='spot',
        interval='1m',
        overwrite=True,
    )
    
    # 下载数据
    downloader.download_klines()
    
    print("\nSuccess!")

