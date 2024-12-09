import datetime
timestamp_ns = 1403636579809262870
dt = datetime.datetime.fromtimestamp(timestamp_ns / 1e9)
print(dt)
