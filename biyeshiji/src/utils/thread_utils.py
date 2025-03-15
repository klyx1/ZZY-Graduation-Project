# 多线程队列/同步工具
import threading
from queue import Queue


class DoubleBuffer:
    """ 双缓冲队列消除多线程数据竞争 """

    def __init__(self):
        self.front = None
        self.back = None
        self.lock = threading.Lock()

    def write(self, data):
        """ 写入后台缓冲区 """
        with self.lock:
            self.back = data

    def swap(self):
        """ 交换前后缓冲区并返回最新数据 """
        with self.lock:
            self.front, self.back = self.back, None
            return self.front


class TimedQueue(Queue):
    """ 带超时机制的安全队列 """

    def get_with_timeout(self, timeout=1.0):
        try:
            return self.get(block=True, timeout=timeout)
        except Exception as e:
            raise TimeoutError(f"队列获取超时: {str(e)}")