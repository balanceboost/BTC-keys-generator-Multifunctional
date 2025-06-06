# BTC Keys Generator
![2](https://github.com/user-attachments/assets/047f8df8-3c11-4646-9bbf-53ed752857dc)
![4](https://github.com/user-attachments/assets/d175f953-5ef2-4260-a543-dc0404c890c0)

A Python script for generating Bitcoin private keys using various methods, including random keys, mnemonic phrases, text-based inputs, and pattern-based approaches. Designed for **educational and research purposes only**, this tool can generate and check Bitcoin addresses against local files or blockchain APIs.

⚠️ **Warning**: Generating and testing private keys is highly sensitive and can lead to loss of funds or legal issues if misused. Use this tool only for learning, testing on Bitcoin testnet, or analyzing historical wallets. Never use generated keys for real Bitcoin transactions without proper security measures. The authors are not responsible for any misuse or consequences.

## Features
**Multiple Generation Methods**:
  1. Standard Random Keys
  2. Vanity Addresses (custom prefix/suffix)
  3. Keys in Specific Numeric Ranges (Groups A-H)
  4. BIP-39 Mnemonic Phrases (12 or 24 words)
  5. Vulnerable Keys (predictable patterns)
  6. Text-Based Keys (passwords, brainwallets, leaked keys, etc.)
  7. Weak RNG Simulation
  8. Pattern-Based Keys (repeating bytes, sequential patterns, etc.)
- **Address Checking**:
  - Compare generated addresses against local text files.
  - Query blockchain APIs (e.g., mempool.space, blockstream.info) for transactions and balances.
  - Support for both file-based and API-based checks.
- **Telegram Notifications**: Send alerts for successful matches or balances (optional).
- **Bloom Filter**: Avoid rechecking duplicate addresses.
- **Logging**: Detailed logs saved to `btc_generator.log`.
- **Interactive CLI**: User-friendly menu with colorized output.
- **State Saving**: Resumes progress after interruptions.

## Prerequisites
- **Python**: 3.8 or higher.
- **Git**: For cloning the repository.
- **Internet Connection**: Required for API-based checks.
- **Operating System**: Tested on Windows, Linux, and macOS.
- **Optional**: Telegram bot token and chat ID for notifications.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/BTC-Keys-Generator.git
   cd BTC-Keys-Generator
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create `API.ini`**:
   Copy the provided `API.ini` template to the project root and configure it:
   ```ini
   [API]
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here

   [ENDPOINTS]
   endpoint_0_name=mempool.space
   endpoint_0_url=https://mempool.space/api/address/{address}
   endpoint_0_txs_url=https://mempool.space/api/address/{address}/txs/chain
   endpoint_0_tx_count_key=chain_stats.tx_count
   endpoint_0_balance_key=chain_stats

   endpoint_1_name=blockstream.info
   endpoint_1_url=https://blockstream.info/api/address/{address}
   endpoint_1_txs_url=https://blockstream.info/api/address/{address}/txs
   endpoint_1_tx_count_key=chain_stats.tx_count
   endpoint_1_balance_key=chain_stats
   ```
   - **Telegram (optional)**: Obtain a bot token from [BotFather](https://t.me/BotFather) and your chat ID.
   - **Endpoints**: Customize API endpoints or use defaults (mempool.space, blockstream.info, blockcypher.com).

2. **Prepare Input Files** (for method 6 or file-based checks):
   - **Address Files**: Text files with one Bitcoin address per line (e.g., `addresses.txt`).
   - **Password/Phrase Files**: Text files with passwords, phrases, or keys (e.g., `rockyou.txt` for passwords, `leaked_keys.txt` for WIF/hex keys).
   - **Format**: One entry per line, UTF-8 encoding.

## Usage

1. **Run the Script**:
   ```bash
   python btc_generator.py
   ```

2. **Follow the Interactive Menu**:
   - **Select Generation Method** (1–8): Choose a method (e.g., 8 for Pattern-Based Keys).
   - **Choose Subcategory** (for method 6): Select from Simple Passwords, Brainwallets, etc.
   - **Select Check Mode**:
     - `file`: Compare addresses against local files.
     - `api`: Query blockchain APIs for transactions/balances.
     - `both`: Use both methods.
   - **Provide Input Files** (if needed): Specify paths to address or password files.
   - **Configure Parameters** (if applicable):
     - Group (A–H) for method 3.
     - Vulnerable type (1–3) for method 5.
     - Prefix/suffix for method 2.
     - Word count (12 or 24) for methods 4 or 6 (subcategory 3).

3. **Monitor Output**:
   - **Console**: Displays progress, statistics, and matches.
   - **Logs**: Saved to `btc_generator.log`.
   - **Results**:
     - Successful wallets (matches or balances) saved to `successful_wallets_btc.txt`.
     - Failed wallets saved to `bad_wallets_btc.txt`.
     - State saved to `state_btc.json` for resuming.

4. **Pause/Resume**:
   - Press **Page Up** or **Page Down** to pause/resume generation.

5. **Example**:
   To generate pattern-based keys (method 8) and check via API:
   - Select method `8`.
   - Choose check mode `api`.
   - The script generates keys using patterns (e.g., repeating bytes, mirrored sequences).
   - Successful keys are logged and saved to `successful_wallets_btc.txt`.

## Methods
1. **Standard Random Keys**: Generates cryptographically secure random keys.
2. **Vanity Addresses**: Finds addresses with custom prefixes/suffixes (slow).
3. **Keys in Groups**: Generates keys in numeric ranges (A–H).
4. **Mnemonic Phrases**: Creates BIP-39 phrases (12/24 words) and derives keys.
5. **Vulnerable Keys**: Generates predictable keys (e.g., repeating bytes).
6. **Text-Based Keys**:
   - Simple Passwords: Hashes passwords with SHA-256.
   - Brainwallets: Hashes phrases with SHA-256/SHA-1.
   - BIP-39 Variants: Generates mnemonic phrases from BIP-39 words.
   - Leaked Keys: Tests WIF/hex keys.
   - Dictionary Phrases: Hashes memorable phrases with SHA-256.
7. **Weak RNG**: Simulates weak random number generators (e.g., time-based seeds).
8. **Pattern-Based Keys**: Generates keys with patterns (e.g., sequential bytes, time-based).

## File Formats
- **Address Files** (for `file` or `both` modes):
  - One Bitcoin address per line (e.g., `1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa`).
  - UTF-8 encoding.
- **Input Files for Method 6**:
  - **Simple Passwords**: Passwords (e.g., `password123`).
  - **Brainwallets/Dictionary Phrases**: Phrases (e.g., `to be or not to be`).
  - **Leaked Keys**: WIF (e.g., `5J...`) or hex (64 characters).
  - **BIP-39 Variants**: No file needed (uses internal BIP-39 wordlist).
- **Output Files**:
  - `successful_wallets_btc.txt`: Successful keys with addresses, balances, etc.
  - `bad_wallets_btc.txt`: Keys with no matches or balance.
  - `state_btc.json`: Progress state.
  - `btc_generator.log`: Detailed logs.

## Troubleshooting
- **API Errors (429, 403)**: Rate limits. The script pauses endpoints automatically. Check `API.ini` for valid endpoints.
- **No Internet**: API checks will fail; use `file` mode or restore connectivity.
- **Invalid Files**: Ensure input files are UTF-8 and formatted correctly.
- **Keyboard Module Issues**: On Linux/macOS, run with `sudo` or install `libevdev` for `keyboard` module.
- **Memory Usage**: Large address files may require significant RAM. Use smaller files or increase Bloom filter capacity.
- **Logs**: Check `btc_generator.log` for detailed errors.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for details.

## Disclaimer
This tool is for **educational and research purposes only**. Misuse, including unauthorized access to Bitcoin wallets, is illegal and unethical. The authors are not liable for any damages or legal consequences resulting from the use of this software.

    To support the author: TFbR9gXb5r6pcALasjX1FKBArbKc4xBjY8

