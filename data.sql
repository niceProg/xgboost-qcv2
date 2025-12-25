--
-- Table structure for table `cg_funding_rate_history`
--

CREATE TABLE `cg_funding_rate_history` (
  `id` bigint(20) NOT NULL,
  `exchange` varchar(50) NOT NULL,
  `pair` varchar(50) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `open` decimal(18,8) DEFAULT NULL,
  `high` decimal(18,8) DEFAULT NULL,
  `low` decimal(18,8) DEFAULT NULL,
  `close` decimal(18,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cg_funding_rate_history`
--

INSERT INTO `cg_funding_rate_history` (`id`, `exchange`, `pair`, `interval`, `time`, `open`, `high`, `low`, `close`, `created_at`, `updated_at`) VALUES
(1, 'Binance', 'BTCUSDT', '1h', 1731708000000, 0.01000000, 0.01000000, 0.01000000, 0.01000000, '2025-11-15 21:22:38', '2025-11-19 05:37:01'),
(2, 'Binance', 'BTCUSDT', '1h', 1731711600000, 0.01000000, 0.01000000, 0.01000000, 0.01000000, '2025-11-15 21:22:38', '2025-11-19 05:37:01'),
(3, 'Binance', 'BTCUSDT', '1h', 1731715200000, 0.01000000, 0.01000000, 0.01000000, 0.01000000, '2025-11-15 21:22:38', '2025-11-19 05:37:01'),
(4, 'Binance', 'BTCUSDT', '1h', 1731718800000, 0.01000000, 0.01000000, 0.01000000, 0.01000000, '2025-11-15 21:22:38', '2025-11-19 05:37:01'),
(5, 'Binance', 'BTCUSDT', '1h', 1731722400000, 0.01000000, 0.01000000, 0.01000000, 0.01000000, '2025-11-15 21:22:38', '2025-11-19 05:37:01');

--
-- Table structure for table `cg_futures_aggregated_ask_bids_history`
--

CREATE TABLE `cg_futures_aggregated_ask_bids_history` (
  `id` bigint(20) NOT NULL,
  `exchange_list` varchar(255) NOT NULL,
  `symbol` varchar(20) NOT NULL,
  `base_asset` varchar(20) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `range_percent` decimal(10,2) NOT NULL,
  `time` bigint(20) NOT NULL,
  `aggregated_bids_usd` decimal(38,8) NOT NULL,
  `aggregated_bids_quantity` decimal(38,8) NOT NULL,
  `aggregated_asks_usd` decimal(38,8) NOT NULL,
  `aggregated_asks_quantity` decimal(38,8) NOT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cg_futures_aggregated_ask_bids_history`
--

INSERT INTO `cg_futures_aggregated_ask_bids_history` (`id`, `exchange_list`, `symbol`, `base_asset`, `interval`, `range_percent`, `time`, `aggregated_bids_usd`, `aggregated_bids_quantity`, `aggregated_asks_usd`, `aggregated_asks_quantity`, `created_at`, `updated_at`) VALUES
(1, 'Binance', 'BTC', 'BTC', '1h', 1.00, 1713952800000, 163863324.15060000, 2483.54500000, 108532914.10060000, 1628.70900000, '2025-12-15 07:58:34', '2025-12-15 07:58:34'),
(2, 'Binance', 'BTC', 'BTC', '1h', 1.00, 1713956400000, 163863324.15060000, 2483.54500000, 108532914.10060000, 1628.70900000, '2025-12-15 07:58:34', '2025-12-15 07:58:34'),
(3, 'Binance', 'BTC', 'BTC', '1h', 1.00, 1713960000000, 134984280.95820000, 2039.36000000, 143951279.62680000, 2152.82200000, '2025-12-15 07:58:34', '2025-12-15 07:58:34'),
(4, 'Binance', 'BTC', 'BTC', '1h', 1.00, 1713963600000, 171904472.81150000, 2606.22400000, 112091565.69250000, 1682.36500000, '2025-12-15 07:58:34', '2025-12-15 07:58:34'),
(5, 'Binance', 'BTC', 'BTC', '1h', 1.00, 1713967200000, 158644849.96450000, 2414.94300000, 98864782.35720000, 1490.67700000, '2025-12-15 07:58:34', '2025-12-15 07:58:34');

--
-- Table structure for table `cg_futures_aggregated_taker_buy_sell_volume_history`
--

CREATE TABLE `cg_futures_aggregated_taker_buy_sell_volume_history` (
  `id` bigint(20) NOT NULL,
  `exchange` varchar(50) NOT NULL,
  `symbol` varchar(20) NOT NULL,
  `base_asset` varchar(20) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `unit` varchar(10) NOT NULL DEFAULT 'usd',
  `time` bigint(20) NOT NULL,
  `aggregated_buy_volume` decimal(38,8) NOT NULL,
  `aggregated_sell_volume` decimal(38,8) NOT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cg_futures_aggregated_taker_buy_sell_volume_history`
--

INSERT INTO `cg_futures_aggregated_taker_buy_sell_volume_history` (`id`, `exchange`, `symbol`, `base_asset`, `interval`, `unit`, `time`, `aggregated_buy_volume`, `aggregated_sell_volume`, `created_at`, `updated_at`) VALUES
(1, 'Binance', 'BTC', 'BTC', '1h', 'usd', 1704247200000, 182021815.75630000, 190587992.52350000, '2025-12-23 01:02:43', '2025-12-23 01:02:43'),
(2, 'Binance', 'BTC', 'BTC', '1h', 'usd', 1704250800000, 93585063.54040000, 100737590.18760000, '2025-12-23 01:02:43', '2025-12-23 01:02:43'),
(3, 'Binance', 'BTC', 'BTC', '1h', 'usd', 1704254400000, 94861447.87100000, 122121641.54510000, '2025-12-23 01:02:43', '2025-12-23 01:02:43'),
(4, 'Binance', 'BTC', 'BTC', '1h', 'usd', 1704258000000, 122596558.27690000, 137113213.46030000, '2025-12-23 01:02:43', '2025-12-23 01:02:43'),
(5, 'Binance', 'BTC', 'BTC', '1h', 'usd', 1704261600000, 169044310.47200000, 122308626.41400000, '2025-12-23 01:02:43', '2025-12-23 01:02:43');

--
-- Table structure for table `cg_futures_basis_history`
--

CREATE TABLE `cg_futures_basis_history` (
  `id` bigint(20) NOT NULL,
  `exchange` varchar(50) NOT NULL,
  `pair` varchar(50) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `open_basis` decimal(18,8) DEFAULT NULL,
  `close_basis` decimal(18,8) DEFAULT NULL,
  `open_change` decimal(18,8) DEFAULT NULL,
  `close_change` decimal(18,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cg_futures_basis_history`
--

INSERT INTO `cg_futures_basis_history` (`id`, `exchange`, `pair`, `interval`, `time`, `open_basis`, `close_basis`, `open_change`, `close_change`, `created_at`, `updated_at`) VALUES
(1, 'Binance', 'BTCUSDT', '1h', 1759644000000, 0.02000000, 0.01790000, 24.99000000, 22.31000000, '2025-11-15 21:28:47', '2025-11-19 05:34:18'),
(2, 'Binance', 'BTCUSDT', '1h', 1759647600000, 0.01790000, 0.01700000, 22.41000000, 21.23000000, '2025-11-15 21:28:47', '2025-11-19 05:34:18'),
(3, 'Binance', 'BTCUSDT', '1h', 1759651200000, 0.01690000, 0.01210000, 21.13000000, 15.09000000, '2025-11-15 21:28:47', '2025-11-19 05:34:18'),
(4, 'Binance', 'BTCUSDT', '1h', 1759654800000, 0.01200000, 0.01830000, 15.00000000, 22.50000000, '2025-11-15 21:28:47', '2025-11-19 05:34:18'),
(5, 'Binance', 'BTCUSDT', '1h', 1759658400000, 0.01830000, 0.03830000, 22.51000000, 47.11000000, '2025-11-15 21:28:47', '2025-11-19 05:34:18');

--
-- Table structure for table `cg_futures_price_history`
--

CREATE TABLE `cg_futures_price_history` (
  `id` bigint(20) NOT NULL,
  `exchange` varchar(50) NOT NULL,
  `symbol` varchar(50) NOT NULL,
  `base_asset` varchar(20) NOT NULL,
  `quote_asset` varchar(20) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `open` decimal(30,10) DEFAULT NULL,
  `high` decimal(30,10) DEFAULT NULL,
  `low` decimal(30,10) DEFAULT NULL,
  `close` decimal(30,10) DEFAULT NULL,
  `volume_usd` decimal(38,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cg_futures_price_history`
--

INSERT INTO `cg_futures_price_history` (`id`, `exchange`, `symbol`, `base_asset`, `quote_asset`, `interval`, `time`, `open`, `high`, `low`, `close`, `volume_usd`, `created_at`, `updated_at`) VALUES
(1, 'Binance', 'BTCUSDT', 'BTC', 'USDT', '15m', 1704067200000, 42313.9000000000, 42535.0000000000, 42289.6000000000, 42532.5000000000, 153438561.44100000, '2025-12-22 18:32:02', '2025-12-22 18:32:02'),
(2, 'Binance', 'BTCUSDT', 'BTC', 'USDT', '15m', 1704068100000, 42532.4000000000, 42603.2000000000, 42449.1000000000, 42458.5000000000, 98737115.90020000, '2025-12-22 18:32:02', '2025-12-22 18:32:02'),
(3, 'Binance', 'BTCUSDT', 'BTC', 'USDT', '15m', 1704069000000, 42458.4000000000, 42485.7000000000, 42386.2000000000, 42474.5000000000, 71461033.36520000, '2025-12-22 18:32:02', '2025-12-22 18:32:02'),
(4, 'Binance', 'BTCUSDT', 'BTC', 'USDT', '15m', 1704069900000, 42474.5000000000, 42527.2000000000, 42449.1000000000, 42503.5000000000, 35493455.28470000, '2025-12-22 18:32:02', '2025-12-22 18:32:02'),
(5, 'Binance', 'BTCUSDT', 'BTC', 'USDT', '15m', 1704070800000, 42503.5000000000, 42510.4000000000, 42462.0000000000, 42497.6000000000, 36122792.54340000, '2025-12-22 18:32:02', '2025-12-22 18:32:02');

--
-- Table structure for table `cg_liquidation_aggregated_history`
--

CREATE TABLE `cg_liquidation_aggregated_history` (
  `id` bigint(20) NOT NULL,
  `symbol` varchar(20) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `aggregated_long_liquidation_usd` decimal(38,8) DEFAULT NULL,
  `aggregated_short_liquidation_usd` decimal(38,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cg_liquidation_aggregated_history`
--

INSERT INTO `cg_liquidation_aggregated_history` (`id`, `symbol`, `interval`, `time`, `aggregated_long_liquidation_usd`, `aggregated_short_liquidation_usd`, `created_at`, `updated_at`) VALUES
(1, 'BTC', '1h', 1759644000000, 171340.95380000, 72890.30270000, '2025-11-15 21:26:01', '2025-11-19 05:18:25'),
(2, 'BTC', '1h', 1759647600000, 174213.45790000, 56122.09890000, '2025-11-15 21:26:01', '2025-11-19 05:18:25'),
(3, 'BTC', '1h', 1759651200000, 84038.54560000, 52272.18510000, '2025-11-15 21:26:01', '2025-11-19 05:18:25'),
(4, 'BTC', '1h', 1759654800000, 9247520.72981000, 112729.69320000, '2025-11-15 21:26:01', '2025-11-19 05:18:25'),
(5, 'BTC', '1h', 1759658400000, 3258846.27410000, 96182.11750000, '2025-11-15 21:26:01', '2025-11-19 05:18:25');

--
-- Table structure for table `cg_long_short_global_account_ratio_history`
--

CREATE TABLE `cg_long_short_global_account_ratio_history` (
  `id` bigint(20) NOT NULL,
  `exchange` varchar(50) NOT NULL,
  `pair` varchar(50) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `global_account_long_percent` decimal(18,8) DEFAULT NULL,
  `global_account_short_percent` decimal(18,8) DEFAULT NULL,
  `global_account_long_short_ratio` decimal(18,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cg_long_short_global_account_ratio_history`
--

INSERT INTO `cg_long_short_global_account_ratio_history` (`id`, `exchange`, `pair`, `interval`, `time`, `global_account_long_percent`, `global_account_short_percent`, `global_account_long_short_ratio`, `created_at`, `updated_at`) VALUES
(1, 'Binance', 'BTCUSDT', '1h', 1759644000000, 37.47000000, 62.53000000, 0.60000000, '2025-11-15 21:24:01', '2025-11-19 05:10:30'),
(2, 'Binance', 'BTCUSDT', '1h', 1759647600000, 36.47000000, 63.53000000, 0.57000000, '2025-11-15 21:24:01', '2025-11-19 05:10:30'),
(3, 'Binance', 'BTCUSDT', '1h', 1759651200000, 36.97000000, 63.03000000, 0.59000000, '2025-11-15 21:24:01', '2025-11-19 05:10:30'),
(4, 'Binance', 'BTCUSDT', '1h', 1759654800000, 37.13000000, 62.87000000, 0.59000000, '2025-11-15 21:24:01', '2025-11-19 05:10:30'),
(5, 'Binance', 'BTCUSDT', '1h', 1759658400000, 37.36000000, 62.64000000, 0.60000000, '2025-11-15 21:24:01', '2025-11-19 05:10:30');

--
-- Table structure for table `cg_long_short_top_account_ratio_history`
--

CREATE TABLE `cg_long_short_top_account_ratio_history` (
  `id` bigint(20) NOT NULL,
  `exchange` varchar(50) NOT NULL,
  `pair` varchar(50) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `top_account_long_percent` decimal(18,8) DEFAULT NULL,
  `top_account_short_percent` decimal(18,8) DEFAULT NULL,
  `top_account_long_short_ratio` decimal(18,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='Top Account Long/Short Ratio History - Endpoint: /api/futures/top-long-short-account-ratio/history';

--
-- Dumping data for table `cg_long_short_top_account_ratio_history`
--

INSERT INTO `cg_long_short_top_account_ratio_history` (`id`, `exchange`, `pair`, `interval`, `time`, `top_account_long_percent`, `top_account_short_percent`, `top_account_long_short_ratio`, `created_at`, `updated_at`) VALUES
(1, 'Binance', 'BTCUSDT', '1h', 1759644000000, 41.56000000, 58.44000000, 0.71000000, '2025-11-15 21:25:03', '2025-11-19 05:13:02'),
(2, 'Binance', 'BTCUSDT', '1h', 1759647600000, 41.17000000, 58.83000000, 0.70000000, '2025-11-15 21:25:03', '2025-11-19 05:13:02'),
(3, 'Binance', 'BTCUSDT', '1h', 1759651200000, 41.81000000, 58.19000000, 0.72000000, '2025-11-15 21:25:03', '2025-11-19 05:13:02'),
(4, 'Binance', 'BTCUSDT', '1h', 1759654800000, 42.06000000, 57.94000000, 0.73000000, '2025-11-15 21:25:03', '2025-11-19 05:13:02'),
(5, 'Binance', 'BTCUSDT', '1h', 1759658400000, 42.74000000, 57.26000000, 0.75000000, '2025-11-15 21:25:03', '2025-11-19 05:13:02');

--
-- Table structure for table `cg_open_interest_aggregated_history`
--

CREATE TABLE `cg_open_interest_aggregated_history` (
  `id` bigint(20) NOT NULL,
  `symbol` varchar(20) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `open` decimal(38,8) DEFAULT NULL,
  `high` decimal(38,8) DEFAULT NULL,
  `low` decimal(38,8) DEFAULT NULL,
  `close` decimal(38,8) DEFAULT NULL,
  `unit` varchar(10) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `cg_open_interest_aggregated_history`
--

INSERT INTO `cg_open_interest_aggregated_history` (`id`, `symbol`, `interval`, `time`, `open`, `high`, `low`, `close`, `unit`, `created_at`, `updated_at`) VALUES
(1, 'BTC', '1h', 1759644000000, 93997080482.00000000, 94329566738.00000000, 93886740083.00000000, 94282105680.00000000, 'usd', '2025-11-15 21:23:26', '2025-11-15 21:33:07'),
(2, 'BTC', '1h', 1759647600000, 94282105680.00000000, 94405020198.00000000, 94041197454.00000000, 94041197454.00000000, 'usd', '2025-11-15 21:23:26', '2025-11-15 21:33:07'),
(3, 'BTC', '1h', 1759651200000, 94041197454.00000000, 94050882792.00000000, 93834976241.00000000, 93914826162.00000000, 'usd', '2025-11-15 21:23:26', '2025-11-15 21:33:07'),
(4, 'BTC', '1h', 1759654800000, 93914826162.00000000, 93919652720.00000000, 92066228640.00000000, 92286408528.00000000, 'usd', '2025-11-15 21:23:26', '2025-11-15 21:33:07'),
(5, 'BTC', '1h', 1759658400000, 92286408528.00000000, 92286408528.00000000, 91915091392.00000000, 92164753945.00000000, 'usd', '2025-11-15 21:23:26', '2025-11-16 01:29:24'),