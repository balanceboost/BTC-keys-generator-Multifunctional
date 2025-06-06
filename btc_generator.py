import asyncio
import datetime
import json
import os
import random
import threading
import logging
import keyboard
import aiohttp
import re
import configparser
import pyfiglet
import hashlib
import tqdm
from pybloom_live import BloomFilter
from colorama import Fore, Style
from termcolor import colored
from bitcoinlib.keys import Key
from bitcoinlib.mnemonic import Mnemonic as BitcoinMnemonic
import secrets
from colorama import init as colorama_init
from collections import deque
import itertools
import struct
import time

UNCHECKED_QUEUE = deque()
UNCHECKED_QUEUE_LOCK = asyncio.Lock()

FAILED_WALLETS = []
FAILED_WALLETS_LOCK = asyncio.Lock()

colorama_init(autoreset=True)

# Настройка логирования с BOM
logging.basicConfig(
    filename='btc_generator.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    filemode='a'
)
logger = logging.getLogger(__name__)

# ASCII-арт
ascii_art = pyfiglet.figlet_format("BTC Keys Generator", font="standard")
colored_art = colored(ascii_art, 'cyan')
print(colored_art)

# Константы
SUCCESS_FILE = 'successful_wallets_btc.txt'
BAD_FILE = 'bad_wallets_btc.txt'
STATE_FILE = 'state_btc.json'     

# Задержки и лимиты
REQUEST_INTERVAL = 0.2
MAX_TIMEOUT = 60
MIN_RETRY_AFTER = 10
MAX_RETRIES = 3
RETRY_BACKOFF = 2
PAUSE_COUNT_LIMIT = 3
PAUSE_WINDOW = 3600

# Чтение конфигурации
def load_config():
    default_endpoints = [
        {
            'name': 'mempool.space',
            'url': 'https://mempool.space/api/address/{address}',
            'txs_url': 'https://mempool.space/api/address/{address}/txs/chain',
            'tx_count_key': 'chain_stats.tx_count',
            'balance_key': 'chain_stats',
            'active': True,
            'limit_reached_time': None,
            'temp_pauses': [],
            'last_used': None
        },
        {
            'name': 'blockstream.info',
            'url': 'https://blockstream.info/api/address/{address}',
            'txs_url': 'https://blockstream.info/api/address/{address}/txs',
            'tx_count_key': 'chain_stats.tx_count',
            'balance_key': 'chain_stats',
            'active': True,
            'limit_reached_time': None,
            'temp_pauses': [],
            'last_used': None
        },
        {
            'name': 'blockcypher.com',
            'url': 'https://api.blockcypher.com/v1/btc/main/addrs/{address}',
            'txs_url': 'https://api.blockcypher.com/v1/btc/main/addrs/{address}?limit=50',
            'tx_count_key': 'n_tx',
            'balance_key': None,
            'active': True,
            'limit_reached_time': None,
            'temp_pauses': [],
            'last_used': None
        }
    ]
    
    if not os.path.exists('API.ini'):
        logger.warning("Файл API.ini не найден, используются стандартные эндпоинты")
        return '', '', default_endpoints
    
    config = configparser.ConfigParser()
    config.read('API.ini')
    
    telegram_token = config['API'].get('TELEGRAM_BOT_TOKEN', '') if 'API' in config else ''
    telegram_chat_id = config['API'].get('TELEGRAM_CHAT_ID', '') if 'API' in config else ''
    
    endpoints = []
    if 'ENDPOINTS' in config:
        i = 0
        while True:
            name_key = f'endpoint_{i}_name'
            if name_key not in config['ENDPOINTS']:
                break
            try:
                endpoint = {
                    'name': config['ENDPOINTS'][name_key],
                    'url': config['ENDPOINTS'][f'endpoint_{i}_url'],
                    'txs_url': config['ENDPOINTS'][f'endpoint_{i}_txs_url'],
                    'tx_count_key': config['ENDPOINTS'][f'endpoint_{i}_tx_count_key'],
                    'balance_key': config['ENDPOINTS'].get(f'endpoint_{i}_balance_key', ''),
                    'active': True,
                    'limit_reached_time': None,
                    'temp_pauses': [],
                    'last_used': None
                }
                if not endpoint['url'] or not endpoint['txs_url']:
                    logger.warning(f"Некорректный эндпоинт {endpoint['name']}: пустой URL")
                else:
                    endpoints.append(endpoint)
                    logger.debug(f"Загружен эндпоинт: {endpoint['name']}")
            except KeyError as e:
                logger.warning(f"Ошибка загрузки эндпоинта {name_key}: отсутствует ключ {e}")
            i += 1
    
    if not endpoints:
        logger.warning("Эндпоинты в API.ini не найдены или некорректны, используются стандартные")
        endpoints = default_endpoints
    else:
        logger.info(f"Загружено {len(endpoints)} эндпоинтов из API.ini")
    
    return telegram_token, telegram_chat_id, endpoints

# Глобальные переменные
pause_event = asyncio.Event()
bloom_filter = BloomFilter(capacity=10_000_000, error_rate=0.001)
bloom_filter_lock = asyncio.Lock()
pause_event.set()
password_index = 0
PUBLIC_ENDPOINTS = []  # Инициализация пустым списком, будет заполнено в main
logger.info("Инициализированы глобальные переменные.")

GROUP_RANGES = {
    'A': (0x1, 0xFFFFFFFF),
    'B': (0x100000000, 0xFFFFFFFF00000000),
    'C': (0x10000000000000000, 0xFFFFFFFF0000000000000000),
    'D': (0x1000000000000000000000000, 0xFFFFFFFF000000000000000000000000),
    'E': (0x10000000000000000000000000000000, 0xFFFFFFFF00000000000000000000000000000000),
    'F': (0x1000000000000000000000000000000000000000, 0xFFFFFFFF0000000000000000000000000000000000000000),
    'G': (0x100000000000000000000000000000000000000000000000, 0xFFFFFFFF000000000000000000000000000000000000000000000000),
    'H': (0x100000000000000000000000000000000000000000000000000000000, 0xFFFFFFFF00000000000000000000000000000000000000000000000000000000),
}

# Статистика
stats = {
    'keys_generated': 0,
    'matches_file': 0,
    'matches_api': 0,
    'addresses_with_balance': 0,
    'start_time': datetime.datetime.now(),
    'group_stats': {group: {'keys': 0, 'matches_file': 0, 'matches_api': 0} for group in GROUP_RANGES}
}

# Отправка сообщения в Telegram
async def send_telegram_message(message, session, telegram_bot_token, telegram_chat_id, method=None, address=None, private_key=None, mnemonic_phrase=None, balance=None, funded_btc=None, last_tx_date=None):
    logger.debug(f"Отправка сообщения в Telegram: {message}")
    if not telegram_bot_token or not telegram_chat_id:
        logger.warning("Telegram bot token или chat ID не указаны, отправка сообщения невозможна")
        return

    formatted_message = f"**Новое уведомление ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})**\n"
    formatted_message += f"Сообщение: {message}\n"
    if method is not None:
        formatted_message += f"Метод: {method}\n"
    if address:
        formatted_message += f"Адрес: {address}\n"
    if private_key:
        formatted_message += f"Приватный ключ: {private_key}\n"
    if mnemonic_phrase and mnemonic_phrase != 'N/A':
        formatted_message += f"Мнемоническая фраза/Пароль: {mnemonic_phrase}\n"
    if balance is not None:
        formatted_message += f"Баланс: {balance} BTC\n"
    if funded_btc is not None:
        formatted_message += f"Общая сумма переводов: {funded_btc} BTC\n"
    if last_tx_date:
        formatted_message += f"Последняя транзакция: {last_tx_date}\n"

    payload = {
        'chat_id': telegram_chat_id,
        'text': formatted_message,
        'parse_mode': 'Markdown'
    }

    async with API_SEMAPHORE:
        try:
            async with session.post(
                f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage",
                json=payload,
                timeout=10
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ошибка отправки сообщения в Telegram: HTTP {response.status}, {error_text}")
                else:
                    logger.info(f"Сообщение успешно отправлено в Telegram: {message}")
        except asyncio.TimeoutError:
            logger.error("Таймаут при отправке сообщения в Telegram")
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка сети при отправке сообщения в Telegram: {e}")
        except Exception as e:
            logger.error(f"Неизвестная ошибка при отправке сообщения в Telegram: {e}")

# Управление паузой
def toggle_pause():
    if pause_event.is_set():
        pause_event.clear()
        print(Fore.CYAN + "Генерация приостановлена. Нажмите Page Up/Down для продолжения." + Style.RESET_ALL)
    else:
        pause_event.set()
        print(Fore.CYAN + "Генерация возобновлена." + Style.RESET_ALL)

def listen_for_pause():
    keyboard.add_hotkey('page down', toggle_pause)
    keyboard.add_hotkey('page up', toggle_pause)

# Работа с состоянием
def load_state():
    if not os.path.exists(STATE_FILE):
        logger.warning(f"Файл {STATE_FILE} не существует, возвращается начальное состояние")
        return {'start_index': 0, 'last_password_index': 0, 'failed_wallets': []}
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
            logger.debug(f"Загружено состояние: {state}")
            return {
                'start_index': state.get('start_index', 0),
                'last_password_index': state.get('last_password_index', 0),
                'failed_wallets': state.get('failed_wallets', [])
            }
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка декодирования {STATE_FILE}: {e}, возвращается начальное состояние")
        return {'start_index': 0, 'last_password_index': 0, 'failed_wallets': []}

def save_state(start_index, last_password_index=0):
    logger.debug(f"Сохранение состояния: start_index={start_index}, last_password_index={last_password_index}")
    state = {'start_index': start_index, 'last_password_index': last_password_index}
    try:
        if not os.path.exists(STATE_FILE):
            logger.debug(f"Создание нового файла {STATE_FILE}")
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                pass
        if not os.access(os.path.dirname(STATE_FILE) or '.', os.W_OK):
            logger.error(f"Нет прав на запись в директорию для {STATE_FILE}")
            return
        logger.debug(f"Запись состояния в {STATE_FILE}")
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Состояние успешно сохранено в {STATE_FILE}: {state}")
    except PermissionError as e:
        logger.error(f"Ошибка доступа при сохранении {STATE_FILE}: {e}")
    except OSError as e:
        logger.error(f"Ошибка файловой системы при сохранении {STATE_FILE}: {e}")
    except Exception as e:
        logger.error(f"Неизвестная ошибка сохранения {STATE_FILE}: {e}")

# Проверка интернета
async def check_internet(session):
    if not hasattr(check_internet, 'last_check'):
        check_internet.last_check = 0
        check_internet.last_result = False

    current_time = datetime.datetime.now().timestamp()
    if current_time - check_internet.last_check < 60:
        logger.debug(f"Используется кэшированный результат проверки интернета: {check_internet.last_result}")
        return check_internet.last_result

    logger.debug("Начало проверки интернет-соединения")
    available_endpoints = [ep for ep in PUBLIC_ENDPOINTS if can_use_endpoint(ep)]
    if not available_endpoints:
        logger.warning("Нет доступных эндпоинтов для проверки интернета")
        check_internet.last_check = current_time
        check_internet.last_result = False
        return False

    for endpoint in available_endpoints:
        try:
            url = endpoint['url'].format(address='1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa')
            async with session.head(url, timeout=5) as response:
                if response.status == 200:
                    logger.debug(f"Интернет доступен, проверка через {endpoint['name']} успешна")
                    check_internet.last_check = current_time
                    check_internet.last_result = True
                    return True
                else:
                    logger.warning(f"Эндпоинт {endpoint['name']} вернул статус {response.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Ошибка проверки интернета через {endpoint['name']}: {e}")
            continue

    logger.error("Интернет-соединение недоступно")
    check_internet.last_check = current_time
    check_internet.last_result = False
    return False

# Проверка эндпоинта
def can_use_endpoint(endpoint):
    now = datetime.datetime.now()
    if endpoint['limit_reached_time'] and now >= endpoint['limit_reached_time']:
        endpoint['active'] = True
        endpoint['limit_reached_time'] = None
        endpoint['temp_pauses'] = []
        logger.info(f"Эндпоинт {endpoint['name']} восстановлен")
    return endpoint['active']

# Проверка временных пауз
def check_temp_pauses(endpoint):
    now = datetime.datetime.now()
    endpoint['temp_pauses'] = [t for t in endpoint['temp_pauses'] if (now - t).total_seconds() <= PAUSE_WINDOW]
    if len(endpoint['temp_pauses']) >= PAUSE_COUNT_LIMIT:
        endpoint['active'] = False
        endpoint['limit_reached_time'] = now + datetime.timedelta(seconds=3600)
        endpoint['temp_pauses'] = []
        logger.warning(f"Эндпоинт {endpoint['name']} отключён на 1 час")
        return True
    return False

# Сравнение адреса с файлом
async def compare_address_with_file(address, addresses_set):
    logger.debug(f"Начало сравнения адреса: {address}")
    try:
        async with asyncio.timeout(5):
            address_lower = address.lower()
            match = address_lower in addresses_set
            if match:
                logger.info(f"Совпадение найдено: {address}")
                print(Fore.GREEN + f"Совпадение найдено: {address}" + Style.RESET_ALL)
            else:
                logger.debug(f"Совпадений для адреса {address} не найдено")
            logger.debug(f"Конец сравнения адреса: {address}, результат: {match}")
            return match
    except asyncio.TimeoutError:
        logger.error(f"Таймаут сравнения адреса: {address}")
        return False
    except Exception as e:
        logger.error(f"Ошибка сравнения адреса {address}: {e}")
        return False

# Генерация ключей
def generate_private_key():
    key = Key(network='bitcoin')
    return key.wif()

def get_address_from_private_key(private_key):
    try:
        key = Key(private_key, network='bitcoin')
        return key.address()
    except Exception:
        return None

def is_valid_btc_address(address):
    try:
        key = Key(address=address, network='bitcoin')
        return bool(key.address())
    except Exception:
        return False

def generate_mnemonic_key(word_count=12):
    entropy = secrets.token_bytes(16 if word_count == 12 else 32)
    mnemo = BitcoinMnemonic("english")
    mnemonic_phrase = mnemo.to_mnemonic(entropy)
    seed = mnemo.to_seed(mnemonic_phrase)
    private_key = Key(seed[:32], network='bitcoin').wif()
    return private_key, mnemonic_phrase

def generate_vulnerable_key(vulnerable_type):
    logger.debug(f"Генерация уязвимого ключа: тип={vulnerable_type}")
    patterns = {
        1: [bytes([random.randint(1, 255)] * 32), bytes([0xAA, 0xBB] * 16)],
        2: [b'\x00' * 28 + os.urandom(4), os.urandom(4) + b'\x00' * 28],
        3: [bytes(range(32)), b'\xDE\xAD\xBE\xEF' * 8]
    }
    try:
        pattern = random.choice(patterns[vulnerable_type])
        try:
            private_key = Key(pattern, network='bitcoin').wif()
            logger.debug(f"Сгенерирован уязвимый ключ: {private_key[:10]}...")
            return private_key
        except Exception as e:
            logger.error(f"Невалидный ключ для паттерна {pattern.hex()}: {e}")
            return None
    except KeyError:
        logger.error(f"Неверный тип уязвимого ключа: {vulnerable_type}")
        return None

def generate_text_based_key(subtype, password=None, input_files=None, word_count=12):
    logger.debug(f"Начало generate_text_based_key: subtype={subtype}, password={password}, input_files={input_files}, word_count={word_count}")
    if not password and not input_files and subtype != 3:
        logger.error("Не указаны ни пароль, ни входные файлы")
        return [(None, None)]

    if subtype == 3:
        mnemo = BitcoinMnemonic("english")
        wordlist = mnemo.wordlist()
        try:
            # Генерируем случайную мнемоническую фразу из word_count слов
            mnemonic_words = [random.choice(wordlist) for _ in range(word_count)]
            mnemonic = " ".join(mnemonic_words)
            logger.debug(f"Сгенерирована мнемоническая фраза: {mnemonic}")

            # Генерируем сид и ключ, игнорируя валидность
            seed = hashlib.pbkdf2_hmac('sha512', mnemonic.encode('utf-8'), b'mnemonic', 2048)
            private_key = Key(seed[:32], network='bitcoin').wif()
            logger.debug(f"Сгенерирован ключ для мнемоники: {private_key[:10]}...")
            return [(private_key, mnemonic)]
        except Exception as e:
            logger.error(f"Ошибка создания ключа из мнемоники: {e}")
            return [(None, None)]

    if not password:
        lines = load_password_dictionary(input_files)
        if not lines:
            logger.error("Входные файлы пусты")
            return [(None, None)]
        password = random.choice(lines)
        logger.debug(f"Выбрана строка: {password}")

    if subtype == 1 or subtype == 5:
        results = []
        try:
            # Проверяем исходный пароль/фразу
            hash_bytes = hashlib.sha256(password.encode()).digest()
            private_key = Key(hash_bytes, network='bitcoin').wif()
            results.append((private_key, password))
            logger.debug(f"Сгенерирован ключ для {password} (subtype={subtype}): {private_key[:10]}...")

            # Проверяем, содержит ли пароль/фразу заглавные буквы
            if any(c.isupper() for c in password):
                normalized_password = password.lower()
                if normalized_password != password:  # Проверяем, что не полностью в нижнем регистре
                    hash_bytes_lower = hashlib.sha256(normalized_password.encode()).digest()
                    private_key_lower = Key(hash_bytes_lower, network='bitcoin').wif()
                    results.append((private_key_lower, normalized_password))
                    logger.debug(f"Сгенерирован ключ для {normalized_password} в нижнем регистре (subtype={subtype}): {private_key_lower[:10]}...")
            return results if results else [(None, None)]
        except Exception as e:
            logger.error(f"Ошибка создания ключа из строки {password} (subtype={subtype}): {e}")
            return [(None, None)]
    elif subtype == 2:
        results = []
        for hash_func in [hashlib.sha256, hashlib.sha1]:
            try:
                # Проверяем исходную фразу
                hash_bytes = hash_func(password.encode()).digest()
                private_key = Key(hash_bytes[:32], network='bitcoin').wif()
                results.append((private_key, password))
                logger.debug(f"Сгенерирован ключ для brainwallet {password} с {hash_func.__name__}: {private_key[:10]}...")

                # Проверяем, содержит ли фраза заглавные буквы
                if any(c.isupper() for c in password):
                    normalized_password = password.lower()
                    if normalized_password != password:
                        hash_bytes_lower = hash_func(normalized_password.encode()).digest()
                        private_key_lower = Key(hash_bytes_lower[:32], network='bitcoin').wif()
                        results.append((private_key_lower, normalized_password))
                        logger.debug(f"Сгенерирован ключ для brainwallet {normalized_password} в нижнем регистре с {hash_func.__name__}: {private_key_lower[:10]}...")
            except Exception as e:
                logger.error(f"Ошибка создания brainwallet ключа для {password} с {hash_func.__name__}: {e}")
        return results if results else [(None, None)]
    elif subtype == 4:
        try:
            private_key = Key(password, network='bitcoin').wif()
            logger.debug(f"Сгенерирован ключ для leaked key {password}: {private_key[:10]}...")
            return [(private_key, password)]
        except Exception:
            logger.debug(f"Невалидный ключ: {password}")
            return [(None, None)]
    logger.error(f"Неизвестная подкатегория: {subtype}")
    return [(None, None)]

def generate_weak_rng_key():
    timestamp = int(datetime.datetime.now().timestamp() * 1000000)
    weak_entropy = secrets.randbits(32)
    seed = (timestamp ^ weak_entropy) & 0xFFFFFFFF
    random.seed(seed + random.getrandbits(16))
    weak_bytes = bytes([random.randint(0, 255) for _ in range(32)])
    try:
        private_key = Key(weak_bytes, network='bitcoin').wif()
        logger.debug(f"Сгенерирован слабый ключ: {private_key[:10]}...")
        return private_key, None
    except Exception as e:
        logger.error(f"Ошибка генерации слабого ключа: {e}")
        return None, None

def generate_pattern_based_key():
    # Список шаблонов для генерации ключей
    patterns = [
        lambda: bytes([random.randint(1, 255)] * 32),  # Повторяющийся случайный байт
        lambda: b'\x00' * 16 + os.urandom(16),  # Половина нулей, половина случайных
        lambda: bytes([i % 256 for i in range(32)]),  # Последовательные байты
        lambda: bytes([0xAB, 0xCD] * 16),  # Чередование двух байтов
        lambda: bytes(list(range(16)) + list(range(15, -1, -1))),  # Зеркальная последовательность
        lambda: bytes([(i + random.randint(1, 255)) % 256 for i in range(32)]),  # Сдвиг случайного байта
        lambda: struct.pack(">32s", int(time.time() * 1000).to_bytes(32, byteorder='big')),  # Ключ на основе времени
        lambda: bytes([random.randint(0, 255) if i % 2 == 0 else 0xAA for i in range(32)]),  # Чередование случайного и фиксированного
        lambda: bytes([min(i * 17 % 255, 255) for i in range(32)]),  # Пилообразный узор
        lambda: bytes(([random.randint(1, 255)] * 2 + [random.randint(0, 255)] * 2) * 8),  # Случайные пары байтов
        lambda: bytes([0xDE, 0xAD, 0xBE, 0xEF] * 8),  # "Магические" числа
        lambda: b'\x42\x42\x42\x42' + os.urandom(28)  # Фиксированный префикс + случайные
    ]

    # Инициализация глобального итератора
    if not hasattr(generate_pattern_based_key, '_pattern_cycle'):
        generate_pattern_based_key._pattern_cycle = itertools.cycle(range(len(patterns)))
    
    # Получаем следующий индекс шаблона
    pattern_idx = next(generate_pattern_based_key._pattern_cycle)
    pattern_lambda = patterns[pattern_idx]
    
    try:
        # Генерируем шаблон
        pattern = pattern_lambda()
        logger.debug(f"Выбран шаблон {pattern_idx + 1}: {pattern.hex()[:20]}...")
        # Создаём приватный ключ
        private_key = Key(pattern, network='bitcoin').wif()
        logger.debug(f"Сгенерирован ключ: {private_key[:10]}...")
        return private_key, None
    except Exception as e:
        logger.error(f"Ошибка генерации ключа для шаблона {pattern_idx + 1}: {e}")
        return None, None

def generate_vanity_address(prefix=None, suffix=None):
    logger.debug(f"Начало генерации Vanity-адреса: prefix={prefix}, suffix={suffix}")
    base58_pattern = r'^[123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]+$'
    if prefix and (len(prefix) > 8 or not re.match(base58_pattern, prefix)):
        logger.error(f"Некорректный префикс: {prefix}. Должен быть ≤8 символов и содержать только Base58.")
        return None, None
    if suffix and (len(suffix) > 8 or not re.match(base58_pattern, suffix)):
        logger.error(f"Некорректный суффикс: {suffix}. Должен быть ≤8 символов и содержать только Base58.")
        return None, None
    if not prefix and not suffix:
        logger.error("Не указаны префикс или суффикс для Vanity-адреса.")
        return None, None

    max_attempts = 10000
    attempt = 0
    while attempt < max_attempts:
        private_key = generate_private_key()
        address = get_address_from_private_key(private_key)
        if not address:
            attempt += 1
            continue
        address_lower = address.lower()
        if ((prefix and address_lower.startswith(prefix.lower())) or 
            (suffix and address_lower.endswith(suffix.lower()))):
            logger.info(f"Найден Vanity-адрес: {address} после {attempt} попыток")
            return private_key, address
        attempt += 1
        if attempt % 1000 == 0:
            logger.debug(f"Попыток для Vanity-адреса: {attempt}")
    
    logger.warning(f"Не удалось найти Vanity-адрес после {max_attempts} попыток")
    return None, None

def generate_private_key_in_group(group):
    start_range, end_range = GROUP_RANGES[group]
    private_key_int = random.randint(start_range, end_range)
    private_key_bytes = private_key_int.to_bytes(32, byteorder='big')
    return Key(private_key_bytes, network='bitcoin').wif()

async def generate_private_key_by_method(method, subtype=None, password=None, input_files=None, vulnerable_type=None, word_count=12, prefix=None, suffix=None, group=None):
    async with bloom_filter_lock:
        if method == 1:
            private_key = generate_private_key()
            return [(private_key, None)]
        elif method == 2:
            private_key, address = generate_vanity_address(prefix=prefix, suffix=suffix)
            if private_key is None:
                return [(None, None)]
            mnemonic_phrase = f"Префикс: {prefix or 'N/A'}, Суффикс: {suffix or 'N/A'}"
            return [(private_key, mnemonic_phrase)]
        elif method == 3:
            private_key = generate_private_key_in_group(group)
            return [(private_key, None)]
        elif method == 4:
            private_key, mnemonic_phrase = generate_mnemonic_key(word_count)
            return [(private_key, mnemonic_phrase)]
        elif method == 5:
            private_key = generate_vulnerable_key(vulnerable_type)
            return [(private_key, None)]
        elif method == 6:
            results = generate_text_based_key(subtype, password, input_files, word_count)
            return results
        elif method == 7:
            private_key, _ = generate_weak_rng_key()
            return [(private_key, None)]
        elif method == 8:
            private_key, _ = generate_pattern_based_key()
            return [(private_key, None)]
        return [(None, None)]

API_SEMAPHORE = asyncio.Semaphore(5)

async def check_transactions(address, session):
    global endpoint_index
    logger.debug(f"Начало проверки транзакций для адреса {address}")
    if not await check_internet(session):
        logger.warning(f"Отсутствует интернет для проверки адреса {address}")
        return None

    available_endpoints = [ep for ep in PUBLIC_ENDPOINTS if can_use_endpoint(ep)]
    if not available_endpoints:
        logger.warning("Нет доступных эндпоинтов для проверки транзакций")
        return None

    for endpoint in available_endpoints:
        name = endpoint['name']
        logger.debug(f"Выбран эндпоинт {name} для адреса {address}")

        for attempt in range(MAX_RETRIES):
            try:
                url = endpoint['url'].format(address=address)
                logger.debug(f"Запрос к {url}, попытка {attempt + 1}")
                async with API_SEMAPHORE:
                    async with session.get(url, timeout=10) as resp:
                        endpoint['last_used'] = datetime.datetime.now()
                        logger.debug(f"Ответ от {name}: статус {resp.status}")
                        if resp.status == 200:
                            try:
                                data = await resp.json()
                                tx_count = data
                                for key in endpoint['tx_count_key'].split('.'):
                                    tx_count = tx_count[key]
                                result = tx_count > 0 if isinstance(tx_count, (int, float)) else len(tx_count) > 0
                                logger.info(f"Эндпоинт {name} для {address}: транзакции={'есть' if result else 'нет'}, tx_count={tx_count}")
                                return result
                            except (KeyError, TypeError, ValueError) as e:
                                logger.error(f"Ошибка парсинга данных от {name} для {address}: {e}")
                                break
                        elif resp.status in (429, 403):
                            logger.warning(f"Эндпоинт {name} вернул {resp.status}, попытка {attempt + 1}/{MAX_RETRIES}")
                            endpoint['temp_pauses'].append(datetime.datetime.now())
                            if check_temp_pauses(endpoint):
                                break
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(min(MIN_RETRY_AFTER * (RETRY_BACKOFF ** attempt), 60))
                            continue
                        else:
                            logger.warning(f"Эндпоинт {name} вернул статус {resp.status}")
                            endpoint['temp_pauses'].append(datetime.datetime.now())
                            if check_temp_pauses(endpoint):
                                break
                            break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Сетевая ошибка {name} для {address}: {e}, попытка {attempt + 1}")
                endpoint['temp_pauses'].append(datetime.datetime.now())
                if check_temp_pauses(endpoint):
                    break
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(min(MIN_RETRY_AFTER * (RETRY_BACKOFF ** attempt), 60))
                continue
            await asyncio.sleep(0.1)

        logger.debug(f"Не удалось проверить транзакции для {address} через {name}, переход к следующему эндпоинту")

    logger.debug(f"Не удалось проверить транзакции для {address} через все эндпоинты")
    return None

async def get_balance(address, session):
    global endpoint_index
    logger.debug(f"Начало проверки баланса для адреса {address}")
    available_endpoints = [ep for ep in PUBLIC_ENDPOINTS if can_use_endpoint(ep)]
    if not available_endpoints:
        logger.warning("Нет доступных эндпоинтов для проверки баланса")
        return None, None, None

    balance_btc = None
    funded_btc = None
    last_tx_date = None

    for endpoint in available_endpoints:
        name = endpoint['name']
        logger.debug(f"Выбран эндпоинт {name} для баланса {address}")

        for attempt in range(MAX_RETRIES):
            try:
                url = endpoint['url'].format(address=address)
                logger.debug(f"Запрос баланса к {url}, попытка {attempt + 1}")
                async with API_SEMAPHORE:
                    async with session.get(url, timeout=10) as resp:
                        endpoint['last_used'] = datetime.datetime.now()
                        logger.debug(f"Ответ баланса от {name}: статус {resp.status}")
                        if resp.status == 200:
                            try:
                                data = await resp.json()
                                if name == 'blockcypher.com':
                                    balance_satoshi = data.get('balance', 0)
                                    balance_btc = balance_satoshi / 1e8
                                    funded_btc = data.get('total_received', 0) / 1e8
                                elif name == 'blockchain.com':
                                    balance_satoshi = data.get('final_balance', 0)
                                    balance_btc = balance_satoshi / 1e8
                                    funded_btc = data.get('total_received', 0) / 1e8
                                elif name == 'bitaps.com':
                                    balance_btc = data.get('data', {}).get('confirmed', {}).get('balance', 0) / 1e8
                                    funded_btc = data.get('data', {}).get('confirmed', {}).get('received_amount', 0) / 1e8
                                elif name == 'chain.so':
                                    balance_btc = float(data.get('data', {}).get('confirmed_balance', '0'))
                                    funded_btc = float(data.get('data', {}).get('total_tx', '0'))
                                elif name == 'btc.com':
                                    balance_satoshi = data.get('data', {}).get('balance', 0)
                                    balance_btc = balance_satoshi / 1e8
                                    funded_btc = data.get('data', {}).get('received', 0) / 1e8
                                else:
                                    chain_stats = data.get('chain_stats', {})
                                    funded = chain_stats.get('funded_txo_sum', 0)
                                    spent = chain_stats.get('spent_txo_sum', 0)
                                    balance_satoshi = funded - spent
                                    balance_btc = balance_satoshi / 1e8
                                    funded_btc = funded / 1e8
                                logger.debug(f"Баланс от {name} для {address}: balance={balance_btc} BTC, funded={funded_btc} BTC")
                            except (KeyError, TypeError, ValueError) as e:
                                logger.error(f"Ошибка парсинга баланса от {name} для {address}: {e}")
                                break
                        elif resp.status in (429, 403):
                            logger.warning(f"Эндпоинт {name} вернул {resp.status}, попытка {attempt + 1}/{MAX_RETRIES}")
                            endpoint['temp_pauses'].append(datetime.datetime.now())
                            if check_temp_pauses(endpoint):
                                break
                            if attempt < MAX_RETRIES - 1:
                                await asyncio.sleep(min(MIN_RETRY_AFTER * (RETRY_BACKOFF ** attempt), 60))
                            continue
                        else:
                            logger.warning(f"Эндпоинт {name} вернул статус {resp.status}")
                            endpoint['temp_pauses'].append(datetime.datetime.now())
                            if check_temp_pauses(endpoint):
                                break
                            break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Сетевая ошибка {name} для {address}: {e}, попытка {attempt + 1}")
                endpoint['temp_pauses'].append(datetime.datetime.now())
                if check_temp_pauses(endpoint):
                    break
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(min(MIN_RETRY_AFTER * (RETRY_BACKOFF ** attempt), 60))
                continue

            try:
                txs_url = endpoint['txs_url'].format(address=address)
                logger.debug(f"Запрос транзакций к {txs_url}")
                async with API_SEMAPHORE:
                    async with session.get(txs_url, timeout=10) as txs_resp:
                        logger.debug(f"Ответ транзакций от {name}: статус {txs_resp.status}")
                        if txs_resp.status == 200:
                            try:
                                txs_data = await txs_resp.json()
                                if name == 'blockcypher.com':
                                    if txs_data.get('txrefs', []) and isinstance(txs_data['txrefs'], list):
                                        last_tx = txs_data['txrefs'][0]
                                        block_time = last_tx.get('confirmed')
                                        if block_time:
                                            last_tx_date = datetime.datetime.strptime(block_time, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M:%S')
                                elif name == 'blockchain.com':
                                    if txs_data.get('txs', []) and isinstance(txs_data['txs'], list):
                                        last_tx = txs_data['txs'][0]
                                        block_time = last_tx.get('time')
                                        if block_time:
                                            last_tx_date = datetime.datetime.fromtimestamp(block_time).strftime('%Y-%m-%d %H:%M:%S')
                                elif name == 'bitaps.com':
                                    if txs_data.get('data', {}).get('txs', []) and isinstance(txs_data['data']['txs'], list):
                                        last_tx = txs_data['data']['txs'][0]
                                        block_time = last_tx.get('timestamp')
                                        if block_time:
                                            last_tx_date = datetime.datetime.fromtimestamp(block_time).strftime('%Y-%m-%d %H:%M:%S')
                                elif name == 'chain.so':
                                    if txs_data.get('data', {}).get('txs', []) and isinstance(txs_data['data']['txs'], list):
                                        last_tx = txs_data['data']['txs'][0]
                                        block_time = last_tx.get('time')
                                        if block_time:
                                            last_tx_date = datetime.datetime.fromtimestamp(int(block_time)).strftime('%Y-%m-%d %H:%M:%S')
                                elif name == 'btc.com':
                                    if txs_data.get('data', []) and isinstance(txs_data['data'], list):
                                        last_tx = txs_data['data'][0]
                                        block_time = last_tx.get('block_time')
                                        if block_time:
                                            last_tx_date = datetime.datetime.fromtimestamp(block_time).strftime('%Y-%m-%d %H:%M:%S')
                                else:
                                    if txs_data and isinstance(txs_data, list) and len(txs_data) > 0:
                                        last_tx = txs_data[0]
                                        block_time = last_tx.get('status', {}).get('block_time')
                                        if block_time:
                                            last_tx_date = datetime.datetime.fromtimestamp(block_time).strftime('%Y-%m-%d %H:%M:%S')
                                logger.debug(f"Дата последней транзакции для {address}: {last_tx_date or 'нет'}")
                            except (KeyError, TypeError, ValueError) as e:
                                logger.error(f"Ошибка парсинга транзакций от {name} для {address}: {e}")
                        else:
                            logger.warning(f"Ошибка ответа транзакций от {name}: статус {txs_resp.status}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Сетевая ошибка транзакций {name} для {address}: {e}")

            if balance_btc is not None:
                logger.info(f"Успешно получены данные для {address}: balance={balance_btc} BTC, funded={funded_btc} BTC, last_tx_date={last_tx_date}")
                return balance_btc, funded_btc, last_tx_date
            await asyncio.sleep(0.1)

        logger.debug(f"Не удалось получить баланс для {address} через {name}, переход к следующему эндпоинту")

    logger.debug(f"Не удалось получить баланс для {address} через все эндпоинты")
    return None, None, None

async def load_addresses_to_memory(file_paths):
    addresses = set()
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"Файл адресов {file_path} не найден.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm.tqdm(f, desc=f"Загрузка {os.path.basename(file_path)}"):
                    line = line.strip()
                    if line and is_valid_btc_address(line):
                        addresses.add(line.lower())
        except Exception as e:
            logger.error(f"Ошибка чтения файла адресов {file_path}: {e}")
    logger.info(f"Загружено {len(addresses)} адресов")
    return addresses

def load_password_dictionary(file_paths):
    passwords = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"Файл {file_path} не найден.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        passwords.append(line)
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
    logger.info(f"Загружено {len(passwords)} строк из файлов")
    return passwords

async def process_wallet(method, private_key, mnemonic_phrase, session, addresses_set, check_mode, group=None):
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, stats
    logger.debug(f"Начало обработки кошелька: private_key={private_key[:10]}..., mnemonic_phrase={mnemonic_phrase}")
    address = None  # Инициализируем address для использования в блоках исключений
    try:
        async with asyncio.timeout(15):  # Увеличено до 15 секунд
            logger.debug(f"Генерация адреса для ключа: {private_key[:10]}...")
            address = get_address_from_private_key(private_key)
            if not address:
                logger.error(f"Не удалось получить адрес для ключа: {private_key[:10]}...")
                async with UNCHECKED_QUEUE_LOCK:
                    UNCHECKED_QUEUE.append((method, private_key, mnemonic_phrase, address or 'unknown', group, 1))
                logger.debug(f"Добавлен в очередь для повторной проверки из-за ошибки генерации адреса: {private_key[:10]}..., попытка 1")
                return

            logger.debug(f"Получен адрес: {address}")
            async with bloom_filter_lock:
                if address.lower() in bloom_filter:
                    logger.debug(f"Адрес {address} уже обработан (найден в Bloom filter)")
                    return
                bloom_filter.add(address.lower())

            stats['keys_generated'] += 1
            if method == 3 and group:
                stats['group_stats'][group]['keys'] += 1

            matched_file = False
            matched_api = False
            balance = None
            funded_btc = None
            last_tx_date = None
            unchecked = False

            if check_mode in ['file', 'both'] and addresses_set:
                logger.debug(f"Проверка адреса {address} по файлу")
                matched_file = await compare_address_with_file(address, addresses_set)
                if matched_file:
                    stats['matches_file'] += 1
                    if method == 3 and group:
                        stats['group_stats'][group]['matches_file'] += 1
                    message = f"Совпадение в файле: {address}"
                    try:
                        with open(SUCCESS_FILE, 'a', encoding='utf-8') as f:
                            f.write(f"Метод: {method}, Адрес: {address}, Приватный ключ: {private_key}, Фраза/Пароль: {mnemonic_phrase or 'N/A'}, Баланс: {balance if balance is not None else 'N/A'} BTC, Общая сумма переводов: {funded_btc if funded_btc is not None else 'N/A'} BTC, Последняя транзакция: {last_tx_date or 'N/A'}\n")
                        logger.info(f"Записан успешный ключ в {SUCCESS_FILE}: {address}")
                        await send_telegram_message(message, session, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, method=method, address=address, private_key=private_key, mnemonic_phrase=mnemonic_phrase)
                    except Exception as e:
                        logger.error(f"Ошибка записи в {SUCCESS_FILE} или отправки Telegram-сообщения: {e}")

            if check_mode in ['api', 'both']:
                logger.debug(f"Начало проверки адреса {address} через API")
                try:
                    has_transactions = await check_transactions(address, session)
                    logger.debug(f"Результат проверки транзакций для {address}: has_transactions={has_transactions}")
                    if has_transactions is None:
                        logger.warning(f"Не удалось проверить транзакции для {address}, помечен как unchecked")
                        unchecked = True
                    elif has_transactions:
                        matched_api = True
                        stats['matches_api'] += 1
                        if method == 3 and group:
                            stats['group_stats'][group]['matches_api'] += 1
                        balance, funded_btc, last_tx_date = await get_balance(address, session)
                        logger.debug(f"Баланс для {address}: balance={balance}, funded={funded_btc}, last_tx_date={last_tx_date}")
                        if balance is not None and balance > 0:
                            stats['addresses_with_balance'] += 1
                        message = f"Адрес с транзакциями: {address}, Баланс: {balance if balance is not None else 'N/A'} BTC, Общая сумма переводов: {funded_btc if funded_btc is not None else 'N/A'} BTC, Последняя транзакция: {last_tx_date or 'N/A'}"
                        try:
                            with open(SUCCESS_FILE, 'a', encoding='utf-8') as f:
                                f.write(f"Метод: {method}, Адрес: {address}, Приватный ключ: {private_key}, Фраза/Пароль: {mnemonic_phrase or 'N/A'}, Баланс: {balance if balance is not None else 'N/A'} BTC, Общая сумма переводов: {funded_btc if funded_btc is not None else 'N/A'} BTC, Последняя транзакция: {last_tx_date or 'N/A'}\n")
                            logger.info(f"Записан успешный ключ в {SUCCESS_FILE}: {address}")
                            await send_telegram_message(message, session, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, method=method, address=address, private_key=private_key, mnemonic_phrase=mnemonic_phrase, balance=balance, funded_btc=funded_btc, last_tx_date=last_tx_date)
                        except Exception as e:
                            logger.error(f"Ошибка записи в {SUCCESS_FILE} или отправки Telegram-сообщения: {e}")
                    else:
                        logger.debug(f"Транзакций для {address} не найдено")
                except Exception as e:
                    logger.error(f"Ошибка проверки API для адреса {address}: {e}")
                    unchecked = True

            logger.debug(f"Классификация ключа: address={address}, matched_file={matched_file}, matched_api={matched_api}, balance={balance}, unchecked={unchecked}")
            if matched_file or matched_api or (balance is not None and balance > 0):
                logger.info(f"Хороший ключ: {address}")
            elif unchecked:
                logger.info(f"Непроверенный ключ: {address}")
                async with UNCHECKED_QUEUE_LOCK:
                    UNCHECKED_QUEUE.append((method, private_key, mnemonic_phrase, address, group, 1))
                logger.debug(f"Добавлен в очередь для повторной проверки: {address}, попытка 1")
            else:
                logger.info(f"Плохой ключ: {address}")
                try:
                    logger.debug(f"Попытка записи плохого ключа в {BAD_FILE}: {address}")
                    if not os.path.exists(BAD_FILE):
                        logger.debug(f"Создание файла {BAD_FILE}")
                        with open(BAD_FILE, 'w', encoding='utf-8') as f:
                            pass
                    if not os.access(os.path.dirname(BAD_FILE) or '.', os.W_OK):
                        logger.error(f"Нет прав на запись в директорию для {BAD_FILE}")
                        return
                    with open(BAD_FILE, 'a', encoding='utf-8') as f:
                        f.write(f"Метод: {method}, Адрес: {address}, Приватный ключ: {private_key}, Фраза/Пароль: {mnemonic_phrase or 'N/A'}\n")
                    logger.info(f"Записан плохой ключ в {BAD_FILE}: {address}")
                except Exception as e:
                    logger.error(f"Ошибка записи в {BAD_FILE}: {e}")

            logger.debug(f"Конец обработки кошелька: address={address}, matched_file={matched_file}, matched_api={matched_api}, balance={balance}, unchecked={unchecked}")
    except asyncio.TimeoutError:
        logger.error(f"Таймаут обработки кошелька: private_key={private_key[:10]}...")
        async with UNCHECKED_QUEUE_LOCK:
            UNCHECKED_QUEUE.append((method, private_key, mnemonic_phrase, address or 'unknown', group, 1))
        logger.debug(f"Добавлен в очередь для повторной проверки из-за таймаута: {address or 'unknown'}, попытка 1")
    except Exception as e:
        logger.error(f"Ошибка обработки кошелька: private_key={private_key[:10]}...: {e}")
        async with UNCHECKED_QUEUE_LOCK:
            UNCHECKED_QUEUE.append((method, private_key, mnemonic_phrase, address or 'unknown', group, 1))
        logger.debug(f"Добавлен в очередь для повторной проверки из-за ошибки: {address or 'unknown'}, попытка 1")

async def retry_unchecked_addresses(session, addresses_set, check_mode):
    MAX_RETRY_ATTEMPTS = 5
    RETRY_INTERVAL = 1800  # 30 минут в секундах
    while True:
        async with UNCHECKED_QUEUE_LOCK:
            if not UNCHECKED_QUEUE:
                await asyncio.sleep(60)
                continue
            method, private_key, mnemonic_phrase, address, group, retry_count = UNCHECKED_QUEUE.popleft()
        logger.debug(f"Повторная проверка адреса: {address}, попытка {retry_count}/{MAX_RETRY_ATTEMPTS}")
        
        if retry_count >= MAX_RETRY_ATTEMPTS:
            logger.info(f"Адрес {address} не проверен после {MAX_RETRY_ATTEMPTS} попыток, сохранён как failed")
            async with FAILED_WALLETS_LOCK:
                FAILED_WALLETS.append({
                    'method': method,
                    'address': address,
                    'private_key': private_key,
                    'mnemonic_phrase': mnemonic_phrase or 'N/A'
                })
            save_state(stats['keys_generated'], password_index)
            continue
        
        await process_wallet(method, private_key, mnemonic_phrase, session, addresses_set, check_mode, group=group)
        await asyncio.sleep(RETRY_INTERVAL)

async def main_loop(method, check_mode, file_paths, group=None, vulnerable_type=None, input_files=None, prefix=None, suffix=None, word_count=12, subtype=None):
    global stats, password_index
    addresses_set = set()
    is_loading_addresses = False
    
    if check_mode in ['file', 'both'] and file_paths:
        is_loading_addresses = True
        addresses_set = await load_addresses_to_memory(file_paths)
        is_loading_addresses = False
        logger.info(f"Размер addresses_set: {len(addresses_set)} адресов")

    passwords = []
    total_passwords = 0
    if method == 6 and subtype != 3 and input_files:
        passwords = load_password_dictionary(input_files)
        total_passwords = len(passwords)
        logger.info(f"Загружено {total_passwords} строк для метода 6")
        if total_passwords == 0:
            logger.error("Входные файлы пусты или не содержат валидных строк")
            return

    state = load_state()
    start_index = state['start_index']
    password_index = state['last_password_index']
    global FAILED_WALLETS
    FAILED_WALLETS = state['failed_wallets']
    logger.info(f"Начало main_loop: start_index={start_index}, password_index={password_index}, failed_wallets={len(FAILED_WALLETS)}")

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
        tasks = []
        progress = tqdm.tqdm(desc="Генерация ключей", unit="keys")
        keys_processed = 0
        retry_task = asyncio.create_task(retry_unchecked_addresses(session, addresses_set, check_mode))
        stats_task = asyncio.create_task(print_stats(is_loading_addresses))
        
        # Выводим пустые строки для резервирования места под статистику
        STATS_LINES = 6
        for _ in range(STATS_LINES):
            print()

        try:
            while True:
                await pause_event.wait()
                if method == 6 and subtype != 3 and password_index >= total_passwords:
                    logger.info(f"Все строки ({total_passwords}) обработаны, password_index={password_index}")
                    save_state(start_index + keys_processed, password_index)
                    break

                batch_size = 100 if method != 2 else 10
                logger.debug(f"Начало обработки батча: password_index={password_index}, batch_size={batch_size}")
                results = []
                try:
                    if method == 6 and subtype != 3 and passwords:
                        for _ in range(batch_size):
                            if password_index >= total_passwords:
                                logger.debug(f"Достигнут конец списка паролей: password_index={password_index}")
                                break
                            password = passwords[password_index]
                            logger.debug(f"Обработка строки: index={password_index}, строка='{password}'")
                            keys = await generate_private_key_by_method(
                                method, subtype=subtype, password=password, input_files=input_files,
                                word_count=word_count
                            )
                            results.extend(keys)
                            password_index += 1
                    else:
                        for _ in range(batch_size):
                            keys = await generate_private_key_by_method(
                                method, subtype=subtype, vulnerable_type=vulnerable_type, input_files=input_files,
                                word_count=word_count, prefix=prefix, suffix=suffix, group=group
                            )
                            results.extend(keys)
                except Exception as e:
                    logger.error(f"Ошибка обработки батча: {e}")
                    save_state(start_index + keys_processed, password_index)
                    continue

                logger.debug(f"Конец обработки батча: password_index={password_index}, результатов={len(results)}")
                valid_results = [(pk, mp) for pk, mp in results if pk is not None]
                logger.debug(f"Валидных ключей в батче: {len(valid_results)}")

                if not valid_results and method != 2:
                    logger.warning("Нет валидных ключей в батче, пропуск обработки")
                    continue

                for private_key, mnemonic_phrase in valid_results:
                    logger.debug(f"Добавление задачи process_wallet для ключа: {private_key[:10]}...")
                    tasks.append(process_wallet(method, private_key, mnemonic_phrase, session, addresses_set, check_mode, group=group))
                    keys_processed += 1
                    progress.update(1)

                if tasks:
                    try:
                        logger.debug(f"Запуск обработки {len(tasks)} задач")
                        for task in tasks:
                            try:
                                await asyncio.wait_for(task, timeout=15 if method == 2 else 10)
                                logger.debug(f"Задача process_wallet для ключа завершена")
                            except asyncio.TimeoutError:
                                logger.error(f"Таймаут выполнения задачи process_wallet")
                            except Exception as e:
                                logger.error(f"Ошибка в задаче process_wallet: {e}")
                        tasks = []
                        logger.debug(f"Завершена обработка всех задач")
                    except Exception as e:
                        logger.error(f"Критическая ошибка обработки задач: {e}")
                    finally:
                        save_state(start_index + keys_processed, password_index if method == 6 and subtype != 3 else 0)
                        logger.info(f"Сохранено состояние: start_index={start_index + keys_processed}, password_index={password_index}")
                else:
                    logger.debug(f"Нет задач для обработки, сохранение состояния")
                    save_state(start_index + keys_processed, password_index if method == 6 and subtype != 3 else 0)
                    logger.info(f"Сохранено состояние: start_index={start_index + keys_processed}, password_index={password_index}")

                await asyncio.sleep(0.01)

        finally:
            retry_task.cancel()
            stats_task.cancel()
            progress.close()

def get_user_input():
    logger.debug("Начало get_user_input")
    print(Fore.CYAN + "=" * 80 + Style.RESET_ALL)
    print(Fore.CYAN + "Выберите метод генерации ключей:" + Style.RESET_ALL)
    print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
    methods = {
        1: {
            "name": "Стандартная генерация",
            "desc": "Генерирует случайные приватные ключи с использованием криптографически безопасного генератора. Подходит для поиска любых ключей без специфичных требований."
        },
        2: {
            "name": "Vanity-адреса",
            "desc": "Создаёт ключи с адресами, начинающимися или заканчивающимися на заданный префикс/суффикс (например, 1Love). Медленный метод из-за поиска совпадений."
        },
        3: {
            "name": "Генерация по группам",
            "desc": "Генерирует ключи в заданных числовых диапазонах (группы A-H). Полезно для поиска ключей в определённых интервалах пространства ключей."
        },
        4: {
            "name": "Мнемонические фразы",
            "desc": "Создаёт стандартные BIP-39 мнемонические фразы (12 или 24 слова) и преобразует их в приватные ключи. Подходит для поиска кошельков с мнемониками."
        },
        5: {
            "name": "Уязвимые ключи",
            "desc": "Генерирует ключи с предсказуемыми или повторяющимися узорами (например, все нули, повторяющиеся байты). Имитирует слабые ключи, созданные ошибочными генераторами."
        },
        6: {
            "name": "Text-Based Key Generation",
            "desc": "Генерирует ключи из текстовых данных (пароли, фразы, ключи) из указанных файлов без мутаций. Включает подкатегории: Simple Passwords, Brainwallets, BIP-39 Variants, Leaked Keys, Dictionary Phrases. Требует текстовые файлы."
        },
        7: {
            "name": "Weak RNG",
            "desc": "Имитирует слабый генератор случайных чисел (например, устаревший seed на основе времени). Подходит для поиска ключей, созданных старыми или ошибочными системами."
        },
        8: {
            "name": "Pattern-Based Keys",
            "desc": "Создаёт ключи на основе предопределённых шаблонов (например, повторяющиеся байты, нули). Полезно для поиска ключей с искусственными узорами."
        }
    }
    for k, v in methods.items():
        print(f"{Fore.YELLOW}{k}{Style.RESET_ALL}: {Fore.CYAN}{v['name']}{Style.RESET_ALL}")
        print(f"    {Fore.WHITE}Описание: {v['desc']}{Style.RESET_ALL}")
    print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
    method = input(Fore.YELLOW + "Введите номер метода (1-8): " + Style.RESET_ALL)
    logger.debug(f"Выбран метод: {method}")
    try:
        method = int(method)
        if method not in methods:
            raise ValueError
    except ValueError:
        print(Fore.RED + "Неверный метод. Используется метод 1." + Style.RESET_ALL)
        logger.warning("Неверный метод, установлен метод 1")
        method = 1

    subtype = None
    bip39_word_file = None
    if method == 6:
        print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
        print(Fore.CYAN + "Выберите подкатегорию для Text-Based Key Generation:" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        subcategories = {
            1: {
                "name": "Simple Passwords",
                "desc": "Генерирует ключи из паролей, хэшированных через SHA-256, без изменений.",
                "file_type": "Текстовый файл с паролями, по одному на строку.",
                "file_examples": "password123, qwerty, admin2023",
                "file_recommendation": "Используйте словари типа rockyou.txt. Подходят пароли длиной 6-20 символов."
            },
            2: {
                "name": "Brainwallets",
                "desc": "Создаёт ключи из фраз, хэшированных через SHA-256 или SHA-1, без изменений.",
                "file_type": "Текстовый файл с фразами или цитатами, по одной на строку.",
                "file_examples": "to be or not to be, correct horse battery staple",
                "file_recommendation": "Используйте файлы с цитатами, текстами песен, строками из книг. Фразы длиной 20-100 символов."
            },
            3: {
                "name": "BIP-39 Variants",
                "desc": "Генерирует мнемонические фразы (12 или 24 слова) из случайных слов, выбранных из словаря BIP-39 без проверки фразы на валидность и создаёт из них приватные ключи.",
                "file_type": "Без файла (используется встроенный словарь)",
                "file_examples": "abandon, ability, zoo",
                "file_recommendation": "Метод подходит для генерации кошельков с использованием слов из словаря BIP-39 без проверки на валидность (кошельки до 2010 года)"
            },
            4: {
                "name": "Leaked Keys",
                "desc": "Проверяет приватные ключи (WIF или hex формат) без изменений.",
                "file_type": "Текстовый файл с приватными ключами в формате WIF или hex, по одному на строку.",
                "file_examples": "5Jxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx, Lxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "file_recommendation": "Используйте базы утёкших ключей. Строки должны быть валидными WIF или 64-символьными hex."
            },
            5: {
                "name": "Dictionary Phrases",
                "desc": "Генерирует ключи из фраз, хэшированных через SHA-256, без изменений.",
                "file_type": "Текстовый файл с запоминающимися фразами, по одной на строку.",
                "file_examples": "liberty or death, the quick brown fox",
                "file_recommendation": "Используйте файлы с цитатами, лозунгами, строками из литературы. Фразы длиной 15-100 символов."
            }
        }
        for k, v in subcategories.items():
            print(f"{Fore.YELLOW}{k}{Style.RESET_ALL}: {Fore.CYAN}{v['name']}{Style.RESET_ALL}")
            print(f"    {Fore.WHITE}Описание: {v['desc']}{Style.RESET_ALL}")
            print(f"    {Fore.GREEN}Тип файла: {v['file_type']}{Style.RESET_ALL}")
            print(f"    {Fore.GREEN}Примеры строк: {v['file_examples']}{Style.RESET_ALL}")
            print(f"    {Fore.GREEN}Рекомендации: {v['file_recommendation']}{Style.RESET_ALL}")
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        subtype = input(Fore.YELLOW + "Введите номер подкатегории (1-5): " + Style.RESET_ALL)
        logger.debug(f"Выбрана подкатегория: {subtype}")
        try:
            subtype = int(subtype)
            if subtype not in subcategories:
                raise ValueError
        except ValueError:
            print(Fore.RED + "Неверная подкатегория. Используется Simple Passwords." + Style.RESET_ALL)
            logger.warning("Неверная подкатегория, установлен Simple Passwords")
            subtype = 1

        if subtype == 3:
            bip39_word_file = None  # Файл не нужен, используем встроенный словарь
            logger.debug("Для подкатегории 3 используется встроенный словарь BIP-39")

    print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
    print(Fore.CYAN + "Выберите режим проверки:" + Style.RESET_ALL)
    print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
    check_modes = {
        1: "Проверка по файлу (file) - сравнивает адреса с локальным файлом",
        2: "Проверка через публичные сервисы (api) - проверяет транзакции через mempool.space, blockstream.info, blockchain.com",
        3: "Проверка и по файлу, и через сервисы (both)"
    }
    for k, v in check_modes.items():
        print(f"{Fore.YELLOW}{k}{Style.RESET_ALL}: {Fore.CYAN}{v}{Style.RESET_ALL}")
    print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
    check_mode = input(Fore.YELLOW + "Введите номер режима (1-3): " + Style.RESET_ALL)
    logger.debug(f"Выбран режим проверки: {check_mode}")
    check_mode_map = {'1': 'file', '2': 'api', '3': 'both'}
    check_mode = check_mode_map.get(check_mode, 'both')

    file_paths = []
    if check_mode in ['file', 'both']:
        print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
        print(Fore.CYAN + "Укажите файлы с адресами для проверки:" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        print(f"{Fore.WHITE}Введите пути к файлам с адресами (через запятую).{Style.RESET_ALL}")
        files_input = input(Fore.YELLOW + "Введите пути к файлам: " + Style.RESET_ALL)
        logger.debug(f"Указаны файлы адресов: {files_input}")
        if files_input.strip():
            file_paths = [f.strip() for f in files_input.split(',')]
            file_paths = [f for f in file_paths if os.path.exists(f)]
            if not file_paths:
                print(Fore.RED + "Указанные файлы не найдены. Режим проверки изменён на 'api'." + Style.RESET_ALL)
                logger.warning("Файлы адресов не найдены, режим изменён на 'api'")
                check_mode = 'api'

    group = None
    if method == 3:
        print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
        print(Fore.CYAN + "Выберите группу (A-H):" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        print(f"{Fore.WHITE}Группы определяют диапазон чисел для генерации ключей (A - маленький, H - большой).{Style.RESET_ALL}")
        group = input(Fore.YELLOW + "Введите группу: " + Style.RESET_ALL).upper()
        logger.debug(f"Выбрана группа: {group}")
        if group not in GROUP_RANGES:
            print(Fore.RED + "Неверная группа. Используется группа A." + Style.RESET_ALL)
            logger.warning("Неверная группа, установлена группа A")
            group = 'A'

    vulnerable_type = None
    if method == 5:
        print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
        print(Fore.CYAN + "Выберите тип уязвимого ключа (1-3):" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        vuln_types = {
            1: "Повторяющиеся байты",
            2: "Нули с случайными байтами",
            3: "Предсказуемые последовательности"
        }
        for k, v in vuln_types.items():
            print(f"{Fore.YELLOW}{k}{Style.RESET_ALL}: {Fore.CYAN}{v}{Style.RESET_ALL}")
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        vulnerable_type = input(Fore.YELLOW + "Введите тип: " + Style.RESET_ALL)
        logger.debug(f"Выбран тип уязвимого ключа: {vulnerable_type}")
        try:
            vulnerable_type = int(vulnerable_type)
            if vulnerable_type not in [1, 2, 3]:
                raise ValueError
        except ValueError:
            print(Fore.RED + "Неверный тип. Используется тип 1." + Style.RESET_ALL)
            logger.warning("Неверный тип уязвимого ключа, установлен тип 1")
            vulnerable_type = 1

    input_files = []
    if method == 6 and subtype != 3:
        while True:
            print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
            print(Fore.CYAN + "Укажите файлы для Text-Based Key Generation:" + Style.RESET_ALL)
            print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
            print(f"{Fore.WHITE}Введите пути к текстовым файлам с паролями, фразами или ключами (через запятую).{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Файлы должны содержать одну строку данных на строку, в формате, подходящем для выбранной подкатегории.{Style.RESET_ALL}")
            files_input = input(Fore.YELLOW + "Введите пути к файлам: " + Style.RESET_ALL)
            logger.debug(f"Указаны входные файлы: {files_input}")
            if files_input.strip():
                input_files = [f.strip() for f in files_input.split(',')]
                input_files = [f for f in input_files if os.path.exists(f)]
                if input_files:
                    break
            print(Fore.RED + "Файлы не указаны или не найдены. Пожалуйста, укажите существующие файлы." + Style.RESET_ALL)
            logger.error("Входные файлы не указаны или не найдены")

    prefix = suffix = None
    if method == 2:
        print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
        print(Fore.CYAN + "Укажите параметры для Vanity-адресов:" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        print(f"{Fore.WHITE}Введите префикс и суффикс для поиска адресов.{Style.RESET_ALL}")
        prefix = input(Fore.YELLOW + "Введите префикс для адреса (например, 1Love): " + Style.RESET_ALL).strip()
        suffix = input(Fore.YELLOW + "Введите суффикс для адреса (например, xAI): " + Style.RESET_ALL).strip()
        logger.debug(f"Vanity-адреса: префикс={prefix}, суффикс={suffix}")

    word_count = 12
    if method in [4, 6] and (method == 4 or subtype == 3):
        print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
        print(Fore.CYAN + "Выберите количество слов в мнемонической фразе:" + Style.RESET_ALL)
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        print(f"{Fore.YELLOW}12{Style.RESET_ALL}: {Fore.CYAN}стандартная фраза, 128 бит энтропии{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}24{Style.RESET_ALL}: {Fore.CYAN}повышенная безопасность, 256 бит энтропии{Style.RESET_ALL}")
        print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
        word_count = input(Fore.YELLOW + "Введите количество слов (12 или 24): " + Style.RESET_ALL)
        logger.debug(f"Выбрано количество слов: {word_count}")
        word_count = 24 if word_count == '24' else 12

    print(Fore.CYAN + "\n" + "=" * 80 + Style.RESET_ALL)
    print(Fore.CYAN + "Сводка выбранных параметров:" + Style.RESET_ALL)
    print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)
    print(f"{Fore.YELLOW}Метод:{Style.RESET_ALL} {Fore.CYAN}{methods[method]['name']}{Style.RESET_ALL}")
    if method == 6:
        print(f"{Fore.YELLOW}Подкатегория:{Style.RESET_ALL} {Fore.CYAN}{subcategories[subtype]['name']}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Режим проверки:{Style.RESET_ALL} {Fore.CYAN}{check_mode}{Style.RESET_ALL}")
    if file_paths:
        print(f"{Fore.YELLOW}Файлы адресов:{Style.RESET_ALL} {Fore.GREEN}{', '.join(file_paths)}{Style.RESET_ALL}")
    if input_files:
        print(f"{Fore.YELLOW}Файлы для метода 6:{Style.RESET_ALL} {Fore.GREEN}{', '.join(input_files)}{Style.RESET_ALL}")
    if bip39_word_file:
        print(f"{Fore.YELLOW}Файл слов BIP-39:{Style.RESET_ALL} {Fore.GREEN}{bip39_word_file}{Style.RESET_ALL}")
    if method == 3:
        print(f"{Fore.YELLOW}Группа:{Style.RESET_ALL} {Fore.CYAN}{group}{Style.RESET_ALL}")
    if method == 5:
        print(f"{Fore.YELLOW}Тип уязвимого ключа:{Style.RESET_ALL} {Fore.CYAN}{vulnerable_type}{Style.RESET_ALL}")
    if method == 2:
        print(f"{Fore.YELLOW}Префикс:{Style.RESET_ALL} {Fore.CYAN}{prefix or 'N/A'}{Style.RESET_ALL}, {Fore.YELLOW}Суффикс:{Style.RESET_ALL} {Fore.CYAN}{suffix or 'N/A'}{Style.RESET_ALL}")
    if method in [4, 6] and (method == 4 or subtype == 3):
        print(f"{Fore.YELLOW}Количество слов в мнемонике:{Style.RESET_ALL} {Fore.CYAN}{word_count}{Style.RESET_ALL}")
    print(Fore.CYAN + "-" * 80 + Style.RESET_ALL)

    logger.debug("Завершение get_user_input")
    return method, check_mode, file_paths, group, vulnerable_type, input_files, prefix, suffix, word_count, subtype, bip39_word_file

async def print_stats(is_loading_addresses=False):
    from colorama import Fore, Style
    import datetime
    
    # Если идёт загрузка адресов, не выводим статистику
    if is_loading_addresses:
        await asyncio.sleep(10)
        return
    
    # Количество строк, которые занимает статистика (для очистки)
    STATS_LINES = 6
    
    while True:
        try:
            # Очистка предыдущих строк
            print(f"\033[{STATS_LINES}A", end="")  # Перемещаем курсор на STATS_LINES строк вверх
            print("\033[K", end="")  # Очищаем строку
            
            elapsed = datetime.datetime.now() - stats['start_time']
            hours, rem = divmod(elapsed.seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Формируем строки статистики
            stats_output = (
                f"{Fore.CYAN}Статистика ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):{Style.RESET_ALL}\n"
                f"{Fore.CYAN}Ключи сгенерированы: {stats['keys_generated']}{Style.RESET_ALL}\n"
                f"{Fore.CYAN}Совпадения по файлу: {stats['matches_file']}{Style.RESET_ALL}\n"
                f"{Fore.CYAN}Совпадения по API: {stats['matches_api']}{Style.RESET_ALL}\n"
                f"{Fore.CYAN}Адреса с балансом: {stats['addresses_with_balance']}{Style.RESET_ALL}\n"
                f"{Fore.CYAN}Время работы: {elapsed_str}{Style.RESET_ALL}\n"
            )
            
            # Выводим статистику
            print(stats_output, end="")
            
            logger.debug(f"Выведена статистика: keys_generated={stats['keys_generated']}, matches_file={stats['matches_file']}, matches_api={stats['matches_api']}")
        except Exception as e:
            logger.error(f"Ошибка вывода статистики: {e}")
        await asyncio.sleep(10)

async def main():
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PUBLIC_ENDPOINTS
    logger.info("Запуск программы")
    threading.Thread(target=listen_for_pause, daemon=True).start()
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PUBLIC_ENDPOINTS = load_config()
    logger.info(f"Загружено {len(PUBLIC_ENDPOINTS)} эндпоинтов")
    
    result = get_user_input()
    if result[0] is None:
        logger.error("Некорректные входные параметры, программа завершена")
        return
    method, check_mode, file_paths, group, vulnerable_type, input_files, prefix, suffix, word_count, subtype, bip39_word_file = result
    logger.info(f"Параметры: метод={method}, режим={check_mode}, файлы={file_paths}, input_files={input_files}, bip39_word_file={bip39_word_file}, группа={group}, уязвимый_тип={vulnerable_type}, префикс={prefix}, суффикс={suffix}, слова={word_count}, подкатегория={subtype}")

    if method == 6 and subtype == 3:
        input_files = []  # Для подкатегории 3 файл не нужен

    try:
        await main_loop(method, check_mode, file_paths, group, vulnerable_type, input_files, prefix, suffix, word_count, subtype)
    except KeyboardInterrupt:
        logger.info(f"Программа остановлена пользователем: keys_generated={stats['keys_generated']}, password_index={password_index}")
        print(Fore.YELLOW + "\nПрограмма остановлена пользователем." + Style.RESET_ALL)
        save_state(stats['keys_generated'], password_index)
    except Exception as e:
        logger.error(f"Критическая ошибка в main: {e}")
        save_state(stats['keys_generated'], password_index)
    finally:
        logger.info(f"Финальное сохранение: keys_generated={stats['keys_generated']}, password_index={password_index}, failed_wallets={len(FAILED_WALLETS)}")
        save_state(stats['keys_generated'], password_index)
        print(Fore.YELLOW + "Программа завершена." + Style.RESET_ALL)

if __name__ == "__main__":
    asyncio.run(main())