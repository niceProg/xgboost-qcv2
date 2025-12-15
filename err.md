(.venv-xgboost) bs000730@temet01bare331:/www/wwwroot/xgboost-qc$ docker-compose logs -f xgboost_pipeline
WARN[0000] /www/wwwroot/xgboost-qc/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion 
xgboost_pipeline  | 2025-12-15 13:48:34,484 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:48:34,484 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:48:34,484 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:48:34,484 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:48:34,484 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:48:34,484 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:48:35,003 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:48:35,003 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:48:35,003 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:48:35,004 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:48:35,004 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:48:35,004 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:48:35,618 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:48:35,618 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:48:35,618 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:48:35,618 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:48:35,618 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:48:35,618 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:48:36,440 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:48:36,440 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:48:36,440 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:48:36,440 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:48:36,440 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:48:36,440 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:48:37,660 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:48:37,660 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:48:37,660 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:48:37,660 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:48:37,660 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:48:37,660 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:48:39,679 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:48:39,679 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:48:39,679 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:48:39,679 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:48:39,679 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:48:39,679 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:48:43,303 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:48:43,303 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:48:43,303 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:48:43,303 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:48:43,303 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:48:43,303 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:48:50,128 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:48:50,128 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:48:50,128 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:48:50,128 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:48:50,128 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:48:50,128 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:49:03,348 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:49:03,348 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:49:03,348 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:49:03,348 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:49:03,348 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:49:03,348 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline  | 2025-12-15 13:49:29,358 - INFO - Database initialized successfully
xgboost_pipeline  | 2025-12-15 13:49:29,358 - INFO - Database storage initialized
xgboost_pipeline  | 2025-12-15 13:49:29,358 - INFO - Starting XGBoost trading pipeline in daily mode
xgboost_pipeline  | 2025-12-15 13:49:29,358 - INFO - Parameters: binance/BTCUSDT/1h
xgboost_pipeline  | 2025-12-15 13:49:29,358 - INFO - Trading Hours: 7:00-16:00 (WIB)
xgboost_pipeline  | 2025-12-15 13:49:29,358 - INFO - Running daily pipeline...
xgboost_pipeline  | Traceback (most recent call last):
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 313, in <module>
xgboost_pipeline  |     main()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 303, in main
xgboost_pipeline  |     success = runner.run(mode=args.mode)
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 267, in run
xgboost_pipeline  |     return self.run_daily_pipeline()
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 176, in run_daily_pipeline
xgboost_pipeline  |     if not self.check_trading_hours():
xgboost_pipeline  |   File "/app/run_daily_pipeline.py", line 91, in check_trading_hours
xgboost_pipeline  |     tz = pytz.timezone(self.timezone) if self.timezone != 'UTC' else None
xgboost_pipeline  |   File "/usr/local/lib/python3.10/site-packages/pytz/__init__.py", line 188, in timezone
xgboost_pipeline  |     raise UnknownTimeZoneError(zone)
xgboost_pipeline  | pytz.exceptions.UnknownTimeZoneError: 'WIB'
xgboost_pipeline exited with code 1 (restarting)