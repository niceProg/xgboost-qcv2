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
-- Table structure for table `cg_futures_aggregated_taker_buy_sell_volume_history`
--

CREATE TABLE `cg_futures_aggregated_taker_buy_sell_volume_history` (
  `id` bigint(20) NOT NULL,
  `exchange_list` varchar(255) NOT NULL,
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