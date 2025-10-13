import time
from functools import partial

import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from typing import Tuple, List
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception
import multiprocessing as mp
from coin_utils.proxy import ProxyPool

"""所有都是以opentime为timestamp"""


def interval_to_seconds(interval):
    seconds_per_unit = {"m": 60, "h": 60 * 60, "d": 24 * 60 * 60, "w": 7 * 24 * 60 * 60}
    return int(interval[:-1]) * seconds_per_unit[interval[-1]]


def interval_to_timedealta(interval):
    dict_ = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "10m": timedelta(minutes=10),
             "15m": timedelta(minutes=15),
             "30m": timedelta(minutes=30), "1h": timedelta(hours=1), "2h": timedelta(minutes=2),
             "4h": timedelta(hours=4)}

    return dict_[interval]


def convert_timestamp_to_str(timestamp_ms):
    """将毫秒级时间戳转换为格式化字符串（UTC时间）"""
    # 转换为秒级时间戳（保留小数位防止精度丢失）
    timestamp_sec = timestamp_ms / 1000.0
    # 转换为UTC时间的datetime对象
    dt = datetime.utcfromtimestamp(timestamp_sec)
    # 格式化为字符串
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# 将想要获取的时间区间分为不同的可以访问的时间对
def get_start_end_pairs(start, end, interval, req_limit):
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    if end is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    start_dt_ts = int(time.mktime(start_dt.timetuple()))
    end_dt_ts = int(time.mktime(end_dt.timetuple()))

    ts_interval = interval_to_seconds(interval)

    res = []
    cur_start = cur_end = start_dt_ts
    while cur_end <= end_dt_ts - ts_interval:
        cur_end = min(end_dt_ts, cur_start + (req_limit - 1) * ts_interval)
        res.append((cur_start, cur_end))
        cur_start = cur_end + ts_interval
    return res


def get_daily_start_end_pairs(start_str, end_str):
    start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
    daily_pairs = []
    current_dt = start_dt
    while current_dt.date() <= end_dt.date():
        day_start_dt = datetime(current_dt.year, current_dt.month, current_dt.day, 0, 0, 0)
        day_end_dt = datetime(current_dt.year, current_dt.month, current_dt.day, 23, 59, 59)
        if day_end_dt > end_dt:
            day_end_dt = end_dt
        day_start_ts = day_start_dt.strftime("%Y-%m-%d %H:%M:%S")  # 使用 .timestamp() 方法更直接
        day_end_ts = day_end_dt.strftime("%Y-%m-%d %H:%M:%S")
        daily_pairs.append((day_start_ts, day_end_ts))
        current_dt += timedelta(days=1)
    return daily_pairs


def compare_intervals(start_ts, end_ts, interval):
    # 计算时间戳之间的差值（单位：秒）
    delta_seconds = end_ts - start_ts
    delta_seconds = delta_seconds.total_seconds()  # 关键修复：转换为秒数

    # 定义单位到秒数的映射
    unit_to_seconds = {
        's': 1,
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }

    # 解析字符串中的数值和单位
    unit = interval[-1]
    num_str = interval[:-1]

    # 转换数值部分为整数或浮点数（例如支持"0.5h"）
    try:
        num = int(num_str)
    except ValueError:
        # 若整数转换失败，尝试浮点数
        num = float(num_str)

    # 计算字符串表示的总秒数
    str_seconds = num * unit_to_seconds.get(unit, 0)

    # 比较时间差与字符串对应的时间
    return delta_seconds > str_seconds


def get_download_ranges(
        target_start,
        target_end,
        existing_start,
        existing_end,
        interval
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    确定需要下载的时间区间

    :return: 需要下载的时间区间列表，格式 [(start1, end1), (start2, end2)]
    """
    target_start = pd.to_datetime(target_start)
    target_end = pd.to_datetime(target_end)

    # 有效性检查
    if target_start > target_end:
        raise ValueError("目标时间范围无效，开始时间晚于结束时间")

    # 当没有现存数据时
    if existing_start == pd.Timestamp.max:
        return [(target_start, target_end)]

    download_ranges = []

    # 前段缺失检查
    if target_start < existing_start and compare_intervals(target_start, existing_start, interval):
        download_ranges.append((target_start, existing_start))

    # 后段缺失检查
    if target_end > existing_end and compare_intervals(existing_end, target_end, interval):
        download_ranges.append((existing_end, target_end))

    return download_ranges


def adjust_timestamp(dt):
    # 提取当前时间的总秒数（忽略日期部分）
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    # 检查是否需要调整
    if (total_seconds + 1) % 300 == 0:
        return dt + timedelta(seconds=1)
    else:
        return dt


@retry(reraise=True, stop=stop_after_attempt(max_attempt_number=10), wait=wait_fixed(5),
       retry=retry_if_exception(Exception))
def req_with_retry(*args, **kwargs):
    return requests.get(*args, **kwargs)

def worker_daily_klines(start_end_lut, symbol_lst, interval, limit, req_interval, proxy_pool):
    proxy = proxy_pool.get_proxy()
    proxy_lut = {'http': proxy} if proxy is not None else None
    # proxy_lut = {'http': f'http://{proxy}', 'https': f'https://{proxy}'} if proxy is not None else None
    if proxy_lut is None:
        symbol_lst = tqdm(symbol_lst)
    for symbol in symbol_lst:
        if proxy_lut is None:
            start_end_lut = tqdm(start_end_lut)
        for day, start_end_pairs in start_end_lut:
            data_dir = f'data/daily/{interval}/{day}/'
            data_path = data_dir + f'{symbol}.pkl'
            if os.path.exists(data_path):
                continue
            daily_res_lst = []
            for since, to in start_end_pairs:
                end_point = "/fapi/v1/continuousKlines"
                params = {
                    'pair': symbol,
                    'contractType': "PERPETUAL",
                    'interval': interval,
                    'startTime': since * 1000,
                    'limit': limit,
                    'endTime': to * 1000
                }
                resp = req_with_retry(SwapData.BASE_URL + end_point, params=params, proxies=proxy_lut)
                daily_res_lst.append(resp.json())
                time.sleep(req_interval)

            data = np.concatenate(daily_res_lst)
            os.makedirs(data_dir, exist_ok=True)
            if not (data.any().item()):
                pd.to_pickle(None, data_path)
                continue
            cols = ["timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "value", "trade_cnt",
                    "active_buy_volume", "active_buy_value"]
            df = pd.DataFrame(np.array(data[:, :-1]), columns=cols)
            df.drop("close_time", axis=1, inplace=True)
            df["timestamp"] = df["timestamp"].apply(
                lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) / 1000)))
            df['mkt_date'] = df['timestamp'].str[:10]
            df['minute'] = df['timestamp'].str[-8:-3].str.replace(':', '')
            del df['timestamp']
            df.set_index(['mkt_date', 'minute'], inplace=True)
            df = df.unstack(level='minute')
            df = df.swaplevel(0, 1, axis=1)
            df = df.sort_index(axis=1)
            df.columns = [f'{col[1]}_{col[0]}' for col in df.columns]
            df = df.reset_index()
            df['instrument'] = symbol
            df.set_index(['mkt_date', 'instrument'], inplace=True)
            df.to_pickle(data_path)
    proxy_pool.put_proxy(proxy)


""" 合约数据现在可以取：
1. k线数据，任意时间间隔， 任意历史时间， klines；
2. 资金费率， 8小时时间间隔，任意历史时间， fundingRate；
3. 账户多空比， 任意时间间隔， 最多过去三十天数据， globalLongShortAccountRatio；
4. 合约主动买卖量， 任意时间间隔，过去三十天数据， takerlongshortRatio， 获取的数据中时间戳为开始时间，已经转化为结束时间；
5. 合约持仓量， 任意时间间隔，过去三十天数据，openInterestHist；
"""


class SwapData:
    save_path = r"../DLStockCombo/data"
    BASE_URL = "https://fapi.binance.com"
    REQ_LIMIT = 1000
    SUPPORT_INTERVAL = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}

    def __init__(self, symbol_lst="BTCUSDT", output=True,
                 datalst=["klines", "fundingRate", "globalLongShortAccountRatio", "takerlongshortRatio"],
                 interval="1m",
                 start="2025-02-10",
                 end="2025-03-09",
                 req_interval=3):
        self.symbol_lst = symbol_lst
        self.output = output
        self.datalst = datalst
        self.interval = interval
        self.start = start
        self.end = end
        self.req_interval = req_interval

    def GetData(self):
        self.output_dict = dict()
        if "klines" in self.datalst:
            self.download_full_klines(start=self.start, end=self.end, req_interval=self.req_interval)
            # df_klines["timestamp"] = df_klines["timestamp"].apply(lambda x: adjust_timestamp(x))
            # self.output_dict["klines"] = df_klines
            # self.output_dict["klines"]["symbol"] = self.symbol_lst
            # self.output_dict["klines"].set_index(keys=["symbol", "timestamp"], inplace=True)
        if "fundingRate" in self.datalst:
            self.output_dict["fundingRate"] = self.download_full_fundingRate(start=self.start, end=self.end,
                                                                             req_interval=self.req_interval)
            self.output_dict["fundingRate"]["symbol"] = self.symbol_lst
            self.output_dict["fundingRate"].set_index(keys=["symbol", "timestamp"], inplace=True)
        if "globalLongShortAccountRatio" in self.datalst:
            self.output_dict["globalLongShortAccountRatio"] = self.download_full_globalLongShortAccountRatio(
                start=self.start, end=self.end, req_interval=self.req_interval)
            self.output_dict["globalLongShortAccountRatio"]["symbol"] = self.symbol_lst
            self.output_dict["globalLongShortAccountRatio"].set_index(keys=["symbol", "timestamp"], inplace=True)
        if "takerlongshortRatio" in self.datalst:
            self.output_dict["takerlongshortRatio"] = self.download_full_takerlongshortRatio(start=self.start,
                                                                                             end=self.end,
                                                                                             req_interval=self.req_interval)
            self.output_dict["takerlongshortRatio"]["symbol"] = self.symbol_lst
            self.output_dict["takerlongshortRatio"].set_index(keys=["symbol", "timestamp"], inplace=True)
        if "openInterestHist" in self.datalst:
            self.output_dict["openInterestHist"] = self.download_full_openInterestHist(start=self.start, end=self.end,
                                                                                       req_interval=self.req_interval)
            self.output_dict["openInterestHist"]["symbol"] = self.symbol_lst
            self.output_dict["openInterestHist"].set_index(keys=["symbol", "timestamp"], inplace=True)

    # 获取k线数据, 默认是永续合约数据
    @retry(reraise=True, stop=stop_after_attempt(max_attempt_number=10), wait=wait_fixed(5),
           retry=retry_if_exception(Exception))
    def get_klines(self, symbol, interval='1h', since=None, limit=1000, to=None):
        end_point = "/fapi/v1/continuousKlines"
        params = {
            'pair': symbol,
            'contractType': "PERPETUAL",
            'interval': interval,
            'startTime': since * 1000,
            'limit': limit,
            'endTime': to * 1000
        }
        resp = requests.get(SwapData.BASE_URL + end_point, params=params)
        return resp.json()

    # 由于最多返回1000条的数据，所以大于1000条数据需要迭代计算
    def download_full_klines(self, start, limit=1500, end=None, req_interval=None):
        # data_save_path = os.path.join(SwapData.save_path, self.symbol_lst, "continuousKlines", self.interval)

        # # 读取已有数据
        # download_ranges = []
        # df_existing = pd.DataFrame()
        # data_parquet = ""
        # if os.path.exists(data_save_path) and (len(os.listdir(data_save_path)) != 0):
        #     data_csv = os.listdir(data_save_path)[0]
        #     try:
        #         df_existing = pd.read_parquet(
        #             os.path.join(data_save_path, data_parquet),
        #             parse_dates=['timestamp'],
        #             date_parser=pd.to_datetime
        #         )
        #
        #         if not df_existing.empty:
        #             df_existing = df_existing.sort_values('timestamp')
        #             existing_start = df_existing['timestamp'].iloc[0]
        #             existing_end = df_existing['timestamp'].iloc[-1]
        #
        #             # 确定需要下载的时间段
        #             download_ranges += get_download_ranges(
        #                 target_start=start,
        #                 target_end=end,
        #                 existing_start=existing_start,
        #                 existing_end=existing_end,
        #                 interval=self.interval
        #             )
        #
        #             # 没有需要下载的数据
        #             if not download_ranges:
        #                 print("K线数据已完整，无需下载")
        #                 df_existing = df_existing.loc[
        #                     (df_existing["timestamp"] <= end) & (df_existing["timestamp"] >= start)]
        #                 return df_existing
        #
        #         else:
        #             # 初始化现有数据
        #             df_existing = pd.DataFrame()
        #
        #     except Exception as e:
        #         print(f"读取现有文件出错，将重新下载全部数据。错误信息：{str(e)}")
        #         df_existing = pd.DataFrame()

        download_ranges = []
        data_parquet = ''
        df_existing = pd.DataFrame()

        if len(download_ranges) != 0:
            start = datetime.strftime(download_ranges[0][0], "%Y-%m-%d %H:%M:%S")
            end = datetime.strftime(download_ranges[0][-1], "%Y-%m-%d %H:%M:%S")

        if self.interval not in SwapData.SUPPORT_INTERVAL:
            raise Exception("interval {} is not support!!!".format(self.interval))
        start_end_pairs = get_start_end_pairs(start, end, self.interval, req_limit=limit)

        start_end_lst = get_daily_start_end_pairs(start, end)
        spec_get_start_end_pairs = partial(get_start_end_pairs, interval=self.interval, req_limit=limit)
        with mp.Manager() as manager:
            with open('proxy.txt') as f:
                proxy_lst = f.read().splitlines()
            proxy_pool = ProxyPool(proxy_lst, manager)
            spec_worker_klines = partial(worker_daily_klines, symbol_lst=self.symbol_lst, interval=self.interval,
                                         limit=limit, req_interval=req_interval, proxy_pool=proxy_pool)
            with mp.Pool(mp.cpu_count()) as p:
                start_end_pairs_lst = list(
                    tqdm(p.starmap(spec_get_start_end_pairs, start_end_lst), total=len(start_end_lst)))
                start_end_lut = list(zip([i[0][:10] for i in start_end_lst], start_end_pairs_lst))
                len_proxy_pool = len(proxy_pool)
                start_end_lut_lst = [[] for _ in range(len_proxy_pool)]
                for i in range(len(start_end_lut)):
                    start_end_lut_lst[i % len_proxy_pool].append(start_end_lut[i])
                list(tqdm(p.imap(spec_worker_klines, start_end_lut_lst)))

        # klines = []
        # for (start_ts, end_ts) in tqdm(start_end_pairs, desc="K线数据正在下载："):
        #     tmp_kline = self.get_klines(self.symbol_lst.replace("/", ""), self.interval, since=start_ts, limit=limit,
        #                                 to=end_ts)
        #     if len(tmp_kline) > 0:
        #         klines.append(tmp_kline)
        #     if req_interval:
        #         time.sleep(req_interval)
        #
        # klines = np.concatenate(klines)
        # data = []
        # cols = ["timestamp", "open", "high", "low", "close", "volume",
        #         "close_time", "value", "trade_cnt",
        #         "active_buy_volume", "active_buy_value"]
        #
        # for i in range(len(klines)):
        #     tmp_kline = klines[i]
        #     data.append(tmp_kline[:-1])
        #
        # df = pd.DataFrame(np.array(data), columns=cols)
        # df.drop("close_time", axis=1, inplace=True)
        # df["timestamp"] = df["timestamp"].apply(
        #     lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) / 1000)))
        #
        # # 合并数据
        # df = pd.concat([df, df_existing], ignore_index=True)
        # df["timestamp"] = pd.to_datetime(df["timestamp"])
        # df = df.sort_values('timestamp')
        # df = df.drop_duplicates(subset=['timestamp'], keep='last')
        #
        # real_start = str(df["timestamp"].iloc[0]).split(" ")[0]
        # real_end = str(df["timestamp"].iloc[-1]).split(" ")[0]
        #
        # if data_parquet != "":
        #     os.remove(os.path.join(data_save_path, data_parquet))
        #
        # if os.path.exists(data_save_path):
        #     pass
        # else:
        #     os.makedirs(data_save_path)
        # save_to = os.path.join(data_save_path, "{}_{}_{}.parquet".format(
        #     self.symbol_lst.replace("/", "-"),
        #     real_start, real_end))
        #
        # df.to_parquet(save_to, index=False)
        #
        # return df

    def get_fundingRate(self, since=None, limit=1000, to=None):
        end_point = "/fapi/v1/fundingRate"
        params = {
            'symbol': self.symbol_lst,
            'startTime': since * 1000,
            'limit': limit,
            'endTime': to * 1000
        }
        resp = requests.get(SwapData.BASE_URL + end_point, params=params)
        return resp.json()

    # 由于最多返回1000条的数据，所以大于1000条数据需要迭代计算, 间隔为8小时
    def download_full_fundingRate(self, start, end=None, req_interval=None):
        data_save_path = os.path.join(SwapData.save_path, self.symbol_lst, "fundingRate", self.interval)

        # 读取已有数据
        download_ranges = []
        df_existing = pd.DataFrame()
        data_csv = ""
        if os.path.exists(data_save_path) and (len(os.listdir(data_save_path)) != 0):
            data_csv = os.listdir(data_save_path)[0]
            try:
                df_existing = pd.read_csv(
                    os.path.join(data_save_path, data_csv),
                    parse_dates=['timestamp'],
                    date_parser=pd.to_datetime
                )

                if not df_existing.empty:
                    df_existing = df_existing.sort_values('timestamp')
                    existing_start = df_existing['timestamp'].iloc[0]
                    existing_end = df_existing['timestamp'].iloc[-1]

                    # 确定需要下载的时间段
                    download_ranges = get_download_ranges(
                        target_start=start,
                        target_end=end,
                        existing_start=existing_start,
                        existing_end=existing_end,
                        interval=self.interval
                    )

                    # 没有需要下载的数据
                    if not download_ranges:
                        print("资金费率数据已完整，无需下载")
                        df_existing = df_existing.loc[
                            (df_existing["timestamp"] <= end) & (df_existing["timestamp"] >= start)]
                        return df_existing

                else:
                    # 初始化现有数据
                    df_existing = pd.DataFrame()

            except Exception as e:
                print(f"读取现有文件出错，将重新下载全部数据。错误信息：{str(e)}")
                df_existing = pd.DataFrame()

        if len(download_ranges) != 0:
            start = datetime.strftime(download_ranges[0][0], "%Y-%m-%d %H:%M:%S")
            end = datetime.strftime(download_ranges[0][-1], "%Y-%m-%d %H:%M:%S")

        # 获取时间对
        start_end_pairs = get_start_end_pairs(start, end, interval="8h", req_limit=SwapData.REQ_LIMIT)

        FundingRates = []
        for (start_ts, end_ts) in tqdm(start_end_pairs, desc="资金费率数据正在下载："):
            FundingRate = self.get_fundingRate(since=start_ts, limit=SwapData.REQ_LIMIT, to=end_ts)
            if len(FundingRate) > 0:
                FundingRates.append(pd.DataFrame(FundingRate))
            if req_interval:
                time.sleep(req_interval)

        FundingRates = pd.concat(FundingRates, axis=0)
        FundingRates.reset_index(inplace=True, drop=True)
        FundingRates["fundingTime"] = FundingRates["fundingTime"].apply(
            lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) / 1000)))
        FundingRates.rename(columns={"fundingTime": "timestamp"}, inplace=True)

        # 合并数据
        FundingRates = pd.concat([FundingRates, df_existing], ignore_index=True)
        FundingRates["timestamp"] = pd.to_datetime(FundingRates["timestamp"])
        FundingRates = FundingRates.sort_values('timestamp')
        FundingRates = FundingRates.drop_duplicates(subset=['timestamp'], keep='last')

        if data_csv != "":
            os.remove(os.path.join(data_save_path, data_csv))

        real_start = str(FundingRates["timestamp"].iloc[0]).split(" ")[0]
        real_end = str(FundingRates["timestamp"].iloc[-1]).split(" ")[0]

        FundingRates.rename(columns={"fundingTime": "timestamp"})

        if os.path.exists(data_save_path):
            pass
        else:
            os.makedirs(data_save_path)
        save_to = os.path.join(data_save_path, "{}_{}_{}.csv".format(
            self.symbol_lst.replace("/", "-"),
            real_start, real_end))

        FundingRates.to_csv(save_to, index=False)

        return FundingRates

    # 合约主动买卖量
    def get_takerlongshortRatio(self, since=None, limit=1000, to=None):
        end_point = "/futures/data/takerlongshortRatio"
        params = {
            'symbol': self.symbol_lst,
            'startTime': since * 1000,
            'period': self.interval,
            'limit': limit,
            'endTime': to * 1000
        }
        resp = requests.get(SwapData.BASE_URL + end_point, params=params)
        return resp.json()

    # 由于最多返回1000条的数据，所以大于1000条数据需要迭代计算, 间隔为8小时
    def download_full_takerlongshortRatio(self, start, limit=500, end=None, req_interval=None):
        data_save_path = os.path.join(SwapData.save_path, self.symbol_lst, "longshortRatio", self.interval)

        # 读取已有数据
        download_ranges = []
        df_existing = pd.DataFrame()
        data_csv = ""
        if os.path.exists(data_save_path) and (len(os.listdir(data_save_path)) != 0):
            data_csv = os.listdir(data_save_path)[0]
            try:
                df_existing = pd.read_csv(
                    os.path.join(data_save_path, data_csv),
                    parse_dates=['timestamp'],
                    date_parser=pd.to_datetime
                )

                if not df_existing.empty:
                    df_existing = df_existing.sort_values('timestamp')
                    existing_start = df_existing['timestamp'].iloc[0]
                    existing_end = df_existing['timestamp'].iloc[-1]

                    # 确定需要下载的时间段
                    download_ranges = get_download_ranges(
                        target_start=start,
                        target_end=end,
                        existing_start=existing_start,
                        existing_end=existing_end,
                        interval=self.interval
                    )

                    # 没有需要下载的数据
                    if not download_ranges:
                        print("合约主动买卖量数据已完整，无需下载")
                        df_existing = df_existing.loc[
                            (df_existing["timestamp"] <= end) & (df_existing["timestamp"] >= start)]
                        return df_existing

                else:
                    # 初始化现有数据
                    df_existing = pd.DataFrame()

            except Exception as e:
                print(f"读取现有文件出错，将重新下载全部数据。错误信息：{str(e)}")
                df_existing = pd.DataFrame()

        if len(download_ranges) != 0:
            start = datetime.strftime(download_ranges[0][0], "%Y-%m-%d %H:%M:%S")
            end = datetime.strftime(download_ranges[0][-1], "%Y-%m-%d %H:%M:%S")

        start_end_pairs = get_start_end_pairs(start, end, interval=self.interval, req_limit=limit)

        longshortRatios = []
        for (start_ts, end_ts) in tqdm(start_end_pairs, desc="合约主动买卖量数据正在下载："):
            longshortRatio = self.get_takerlongshortRatio(since=start_ts, limit=limit,
                                                          to=end_ts)
            if len(longshortRatio) > 0:
                longshortRatios.append(pd.DataFrame(longshortRatio))
            if req_interval:
                time.sleep(req_interval)

        longshortRatios = pd.concat(longshortRatios, axis=0)
        longshortRatios.reset_index(inplace=True, drop=True)
        longshortRatios["timestamp"] = longshortRatios["timestamp"].apply(
            lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) / 1000)))

        # 合并数据
        longshortRatios["timestamp"] = pd.to_datetime(longshortRatios["timestamp"])
        longshortRatios["timestamp"] = longshortRatios["timestamp"].apply(
            lambda x: x + interval_to_timedealta(self.interval))
        longshortRatios = pd.concat([longshortRatios, df_existing], ignore_index=True)
        longshortRatios["timestamp"] = pd.to_datetime(longshortRatios["timestamp"])
        longshortRatios = longshortRatios.sort_values('timestamp')
        longshortRatios = longshortRatios.drop_duplicates(subset=['timestamp'], keep='last')

        real_start = str(longshortRatios["timestamp"].iloc[0]).split(" ")[0]
        real_end = str(longshortRatios["timestamp"].iloc[-1]).split(" ")[0]

        if data_csv != "":
            os.remove(os.path.join(data_save_path, data_csv))

        if os.path.exists(data_save_path):
            pass
        else:
            os.makedirs(data_save_path)
        save_to = os.path.join(data_save_path, "{}_{}_{}.csv".format(
            self.symbol_lst.replace("/", "-"),
            real_start, real_end))
        longshortRatios.to_csv(save_to, index=False)
        return longshortRatios

    # 多空持仓人数比
    def get_globalLongShortAccountRatio(self, interval='1h', since=None, limit=500, to=None):
        end_point = "/futures/data/globalLongShortAccountRatio"
        params = {
            'symbol': self.symbol_lst,
            'period': interval,
            'startTime': since * 1000,
            'limit': limit,
            'endTime': to * 1000
        }
        resp = requests.get(SwapData.BASE_URL + end_point, params=params)
        return resp.json()

    def download_full_globalLongShortAccountRatio(self, start, end=None, req_interval=None, limit=500):
        data_save_path = os.path.join(SwapData.save_path, self.symbol_lst, "LongShortAccountRatio", self.interval)

        # 读取已有数据
        download_ranges = []
        df_existing = pd.DataFrame()
        data_csv = ""
        if os.path.exists(data_save_path) and (len(os.listdir(data_save_path)) != 0):
            data_csv = os.listdir(data_save_path)[0]
            try:
                df_existing = pd.read_csv(
                    os.path.join(data_save_path, data_csv),
                    parse_dates=['timestamp'],
                    date_parser=pd.to_datetime
                )

                if not df_existing.empty:
                    df_existing = df_existing.sort_values('timestamp')
                    existing_start = df_existing['timestamp'].iloc[0]
                    existing_end = df_existing['timestamp'].iloc[-1]

                    # 确定需要下载的时间段
                    download_ranges = get_download_ranges(
                        target_start=start,
                        target_end=end,
                        existing_start=existing_start,
                        existing_end=existing_end,
                        interval=self.interval
                    )

                    # 没有需要下载的数据
                    if not download_ranges:
                        print("多空持仓人数比数据已完整，无需下载")
                        df_existing = df_existing.loc[
                            (df_existing["timestamp"] <= end) & (df_existing["timestamp"] >= start)]
                        return df_existing

                else:
                    # 初始化现有数据
                    df_existing = pd.DataFrame()

            except Exception as e:
                print(f"读取现有文件出错，将重新下载全部数据。错误信息：{str(e)}")
                df_existing = pd.DataFrame()

        if len(download_ranges) != 0:
            start = datetime.strftime(download_ranges[0][0], "%Y-%m-%d %H:%M:%S")
            end = datetime.strftime(download_ranges[0][-1], "%Y-%m-%d %H:%M:%S")

        start_end_pairs = get_start_end_pairs(start, end, interval=self.interval, req_limit=limit)

        LongShortAccountRatios = []
        for (start_ts, end_ts) in tqdm(start_end_pairs, desc="多空持仓人数比数据正在下载："):
            LongShortAccountRatio = self.get_globalLongShortAccountRatio(interval=self.interval, since=start_ts,
                                                                         limit=limit, to=end_ts)
            if len(LongShortAccountRatio) > 0:
                LongShortAccountRatios.append(pd.DataFrame(LongShortAccountRatio))
            if req_interval:
                time.sleep(req_interval)

        LongShortAccountRatios = pd.concat(LongShortAccountRatios, axis=0)
        LongShortAccountRatios.reset_index(inplace=True, drop=True)
        LongShortAccountRatios["timestamp"] = LongShortAccountRatios["timestamp"].apply(
            lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) / 1000)))

        # 合并数据
        LongShortAccountRatios = pd.concat([LongShortAccountRatios, df_existing], ignore_index=True)
        LongShortAccountRatios["timestamp"] = pd.to_datetime(LongShortAccountRatios["timestamp"])
        LongShortAccountRatios = LongShortAccountRatios.sort_values('timestamp')
        LongShortAccountRatios = LongShortAccountRatios.drop_duplicates(subset=['timestamp'], keep='last')

        real_start = str(LongShortAccountRatios["timestamp"].iloc[0]).split(" ")[0]
        real_end = str(LongShortAccountRatios["timestamp"].iloc[-1]).split(" ")[0]

        if data_csv != "":
            os.remove(os.path.join(data_save_path, data_csv))

        if os.path.exists(data_save_path):
            pass
        else:
            os.makedirs(data_save_path)
        save_to = os.path.join(data_save_path, "{}_{}_{}.csv".format(
            self.symbol_lst.replace("/", "-"),
            real_start, real_end))

        LongShortAccountRatios.to_csv(save_to, index=False)

        return LongShortAccountRatios

    # 合约持仓量
    def get_openInterestHist(self, interval='1h', since=None, limit=500, to=None):
        end_point = "/futures/data/openInterestHist"
        params = {
            'symbol': self.symbol_lst,
            'period': interval,
            'startTime': since * 1000,
            'limit': limit,
            'endTime': to * 1000
        }
        resp = requests.get(SwapData.BASE_URL + end_point, params=params)
        return resp.json()

    def download_full_openInterestHist(self, start, end=None, req_interval=None, limit=500):
        data_save_path = os.path.join(SwapData.save_path, self.symbol_lst, "openInterestHist", self.interval)

        # 读取已有数据
        download_ranges = []
        df_existing = pd.DataFrame()
        data_csv = ""
        if os.path.exists(data_save_path) and (len(os.listdir(data_save_path)) != 0):
            data_csv = os.listdir(data_save_path)[0]
            try:
                df_existing = pd.read_csv(
                    os.path.join(data_save_path, data_csv),
                    parse_dates=['timestamp'],
                    date_parser=pd.to_datetime
                )

                if not df_existing.empty:
                    df_existing = df_existing.sort_values('timestamp')
                    existing_start = df_existing['timestamp'].iloc[0]
                    existing_end = df_existing['timestamp'].iloc[-1]

                    # 确定需要下载的时间段
                    download_ranges = get_download_ranges(
                        target_start=start,
                        target_end=end,
                        existing_start=existing_start,
                        existing_end=existing_end,
                        interval=self.interval
                    )

                    # 没有需要下载的数据
                    if not download_ranges:
                        print("合约持仓量数据已完整，无需下载")
                        df_existing = df_existing.loc[
                            (df_existing["timestamp"] <= end) & (df_existing["timestamp"] >= start)]
                        return df_existing

                else:
                    # 初始化现有数据
                    df_existing = pd.DataFrame()

            except Exception as e:
                print(f"读取现有文件出错，将重新下载全部数据。错误信息：{str(e)}")
                df_existing = pd.DataFrame()

        if len(download_ranges) != 0:
            start = datetime.strftime(download_ranges[0][0], "%Y-%m-%d %H:%M:%S")
            end = datetime.strftime(download_ranges[0][-1], "%Y-%m-%d %H:%M:%S")

        start_end_pairs = get_start_end_pairs(start, end, interval=self.interval, req_limit=limit)

        openInterestHists = []
        for (start_ts, end_ts) in tqdm(start_end_pairs, desc="合约持仓量数据正在下载："):
            openInterestHist = self.get_openInterestHist(interval=self.interval, since=start_ts, limit=limit, to=end_ts)
            if len(openInterestHist) > 0:
                openInterestHists.append(pd.DataFrame(openInterestHist))
            if req_interval:
                time.sleep(req_interval)

        openInterestHists = pd.concat(openInterestHists, axis=0)
        openInterestHists.reset_index(inplace=True, drop=True)
        openInterestHists["timestamp"] = openInterestHists["timestamp"].apply(
            lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(x) / 1000)))

        # 合并数据
        openInterestHists = pd.concat([openInterestHists, df_existing], ignore_index=True)
        openInterestHists["timestamp"] = pd.to_datetime(openInterestHists["timestamp"])
        openInterestHists = openInterestHists.sort_values('timestamp')
        openInterestHists = openInterestHists.drop_duplicates(subset=['timestamp'], keep='last')

        real_start = str(openInterestHists["timestamp"].iloc[0]).split(" ")[0]
        real_end = str(openInterestHists["timestamp"].iloc[-1]).split(" ")[0]

        if data_csv != "":
            os.remove(os.path.join(data_save_path, data_csv))

        if os.path.exists(data_save_path):
            pass
        else:
            os.makedirs(data_save_path)
        save_to = os.path.join(data_save_path, "{}_{}_{}.xsv".format(
            self.symbol_lst.replace("/", "-"),
            real_start, real_end))

        openInterestHists.to_csv(save_to, index=False)
        return openInterestHists


if __name__ == "__main__":
    # Spot_1 = SpotData()
    # df = Spot_1.download_full_klines(interval="15m", start="2024-10-01", req_interval=3)

    Swap_1 = SwapData(symbol_lst=['ETHUSDT', 'BTCUSDT', 'SOLUSDT'], interval='1m', datalst=["klines"], start="2010-01-01 00:00:00",
                      end="2025-01-01 00:00:00", req_interval=4)
    Swap_1.GetData()
    # Swap_1.download_full_klines(start="2025-03-01 00:00:00", end="2025-03-09 00:00:00", req_interval=3)
    # Swap_1.download_full_fundingRate(start="2024-09-01 00:00:00", end="2025-03-09 00:00:00", req_interval=3)
    # Swap_1.download_full_globalLongShortAccountRatio(start="2025-02-10 00:00:00", end="2025-03-09 16:15:00")
    # Swap_1.download_full_takerlongshortRatio(start="2025-02-10 00:00:00", end="2025-03-09 16:15:00")
    # Swap_1.download_full_openInterestHist(start="2025-02-10 00:00:00", end="2025-03-09 16:15:00")

    print("Success!")
