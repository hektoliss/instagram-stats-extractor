import base64
import json
import logging
import os
from io import BytesIO
from typing import Union, Optional

import cv2
import math
import openai
from PIL import Image

# Настройка логирования
logger = logging.getLogger(__name__)

# Настраиваем логирование только если оно еще не настроено
root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('instagram_extractor.log', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=False
    )
else:
    # Если логирование уже настроено, добавляем только файловый handler для этого модуля
    if not any(isinstance(h, logging.FileHandler) and 
               hasattr(h, 'baseFilename') and 
               'instagram_extractor.log' in h.baseFilename 
               for h in logger.handlers):
        file_handler = logging.FileHandler('instagram_extractor.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)


# -----------------------------
# Основная функция
# -----------------------------
def extract_instagram_stats(
    file_path: str,
    api_key: Optional[str] = None,
    timeout: Optional[int] = 60,
    proxy: Optional[str] = None,
) -> dict:
    """
    Главная функция для извлечения статистики поста Instagram из видео или скриншота.
    
    Args:
        file_path: Путь к видео файлу или изображению
        api_key: OpenAI API ключ (опционально, по умолчанию берется из переменной окружения OPENAI_API_KEY)
        timeout: Таймаут запроса в секундах
        proxy: Прокси для запросов (опционально)
    
    Returns:
        Словарь с извлеченной статистикой в фиксированном формате
    
    Raises:
        ValueError: Если API ключ не указан и не найден в переменных окружения
    """
    logger.info(f"Начало анализа файла: {file_path}")
    
    # Получаем API ключ из переменных окружения, если не передан явно
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = (
                "OpenAI API ключ не найден. Укажите его через параметр api_key "
                "или установите переменную окружения OPENAI_API_KEY"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        logger.debug("API ключ передан явно")
    
    # Настройка клиента OpenAI с поддержкой timeout и proxy
    client_kwargs = {"api_key": api_key}
    
    # Настройка timeout и proxy через httpx
    try:
        import httpx
        
        http_client_kwargs = {}
        if timeout:
            http_client_kwargs["timeout"] = httpx.Timeout(timeout, connect=10.0)
        if proxy:
            http_client_kwargs["proxies"] = proxy
        
        # Создаем httpx клиент
        if http_client_kwargs:
            http_client = httpx.Client(**http_client_kwargs)
            client_kwargs["http_client"] = http_client
            logger.debug(f"Настроен HTTP клиент с timeout={timeout}, proxy={proxy is not None}")
    except ImportError:
        logger.warning("httpx не установлен, timeout и proxy могут не работать. Установите: pip install httpx")
    except Exception as e:
        logger.warning(f"Ошибка при настройке HTTP клиента: {e}. Продолжаем без timeout/proxy")
    
    client = openai.OpenAI(**client_kwargs)

    # Определяем тип файла
    is_video = _is_video_file(file_path)
    logger.info(f"Тип файла: {'видео' if is_video else 'изображение'}")
    
    if is_video:
        logger.info("Начало извлечения кадров из видео")
        frames = extract_frames(file_path=file_path)
        if not frames:
            error_msg = f"Не удалось извлечь кадры из видео: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"Извлечено кадров: {len(frames)}")
        encoded_strings = encode_frames(frames=frames)
    else:
        # Для изображений используем сам файл
        logger.info("Открытие изображения")
        try:
            image = Image.open(file_path)
            encoded_strings = [encode_image(image)]
            logger.info("Изображение успешно загружено и закодировано")
        except Exception as e:
            error_msg = f"Ошибка при открытии изображения {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    # Получаем промпт для извлечения статистики
    prompt = get_instagram_stats_prompt()

    # Анализируем изображения с помощью OpenAI
    response = analyze_images_with_openai(
        client=client,
        encoded_strings=encoded_strings,
        prompt=prompt,
        gpt_model="gpt-4o",
    )

    # Парсим JSON из ответа
    try:
        logger.info("Парсинг ответа от GPT")
        stats = parse_json_response(response)
        logger.info("Анализ успешно завершен")
        return stats
    except Exception as e:
        error_msg = f"Не удалось распарсить ответ от GPT: {e}\nОтвет: {response}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


def extract_instagram_stats_from_multiple_files(
    file_paths: list[str],
    api_key: Optional[str] = None,
    timeout: Optional[int] = 60,
    proxy: Optional[str] = None,
) -> dict:
    """
    Главная функция для извлечения статистики поста Instagram из нескольких файлов (скриншотов одного поста).
    
    Args:
        file_paths: Список путей к видео файлам или изображениям
        api_key: OpenAI API ключ (опционально, по умолчанию берется из переменной окружения OPENAI_API_KEY)
        timeout: Таймаут запроса в секундах
        proxy: Прокси для запросов (опционально)
    
    Returns:
        Словарь с извлеченной статистикой в фиксированном формате
    
    Raises:
        ValueError: Если API ключ не указан и не найден в переменных окружения
    """
    logger.info(f"Начало анализа {len(file_paths)} файлов как скриншотов одного поста")
    
    # Получаем API ключ из переменных окружения, если не передан явно
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = (
                "OpenAI API ключ не найден. Укажите его через параметр api_key "
                "или установите переменную окружения OPENAI_API_KEY"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    else:
        logger.debug("API ключ передан явно")
    
    # Настройка клиента OpenAI с поддержкой timeout и proxy
    client_kwargs = {"api_key": api_key}
    
    # Настройка timeout и proxy через httpx
    try:
        import httpx
        
        http_client_kwargs = {}
        if timeout:
            http_client_kwargs["timeout"] = httpx.Timeout(timeout, connect=10.0)
        if proxy:
            http_client_kwargs["proxies"] = proxy
        
        # Создаем httpx клиент
        if http_client_kwargs:
            http_client = httpx.Client(**http_client_kwargs)
            client_kwargs["http_client"] = http_client
            logger.debug(f"Настроен HTTP клиент с timeout={timeout}, proxy={proxy is not None}")
    except ImportError:
        logger.warning("httpx не установлен, timeout и proxy могут не работать. Установите: pip install httpx")
    except Exception as e:
        logger.warning(f"Ошибка при настройке HTTP клиента: {e}. Продолжаем без timeout/proxy")
    
    client = openai.OpenAI(**client_kwargs)
    
    # Обрабатываем все файлы и собираем все изображения
    all_encoded_strings = []
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        logger.info(f"Обработка файла: {file_name}")
        
        # Определяем тип файла
        is_video = _is_video_file(file_path)
        logger.debug(f"Тип файла {file_name}: {'видео' if is_video else 'изображение'}")
        
        if is_video:
            logger.info(f"Начало извлечения кадров из видео: {file_name}")
            frames = extract_frames(file_path=file_path)
            if not frames:
                logger.warning(f"Не удалось извлечь кадры из видео: {file_path}, пропускаем")
                continue
            logger.info(f"Извлечено кадров из {file_name}: {len(frames)}")
            encoded_strings = encode_frames(frames=frames)
            all_encoded_strings.extend(encoded_strings)
        else:
            # Для изображений используем сам файл
            logger.info(f"Открытие изображения: {file_name}")
            try:
                image = Image.open(file_path)
                encoded_string = encode_image(image)
                all_encoded_strings.append(encoded_string)
                logger.info(f"Изображение {file_name} успешно загружено и закодировано")
            except Exception as e:
                error_msg = f"Ошибка при открытии изображения {file_path}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                # Продолжаем обработку других файлов
                continue
    
    if not all_encoded_strings:
        error_msg = "Не удалось обработать ни один из файлов"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Всего подготовлено изображений для анализа: {len(all_encoded_strings)}")
    
    # Получаем промпт для извлечения статистики (с указанием, что это несколько скриншотов)
    prompt = get_instagram_stats_prompt_multiple()
    
    # Анализируем все изображения с помощью OpenAI в одном запросе
    response = analyze_images_with_openai(
        client=client,
        encoded_strings=all_encoded_strings,
        prompt=prompt,
        gpt_model="gpt-4o",
    )
    
    # Парсим JSON из ответа
    try:
        logger.info("Парсинг ответа от GPT")
        stats = parse_json_response(response)
        logger.info("Анализ всех файлов успешно завершен")
        return stats
    except Exception as e:
        error_msg = f"Не удалось распарсить ответ от GPT: {e}\nОтвет: {response}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)


def _is_video_file(file_path: str) -> bool:
    """Проверяет, является ли файл видео."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions


# -----------------------------
# Нарезка кадров (из оригинального кода)
# -----------------------------
def extract_frames(file_path: str):
    try:
        if not os.path.exists(file_path):
            error_msg = f"Не удалось извлечь кадры, файл не существует: `{file_path}`!"
            logger.error(error_msg)
            print(error_msg)
            return None

        logger.debug(f"Открытие видео файла: {file_path}")
        vidcap = cv2.VideoCapture(file_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # Получаем частоту кадров в секунду
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)  # Общее количество кадров

        if not fps:
            error_msg = f"Ошибка расчета fps: {fps}"
            logger.error(error_msg)
            print(error_msg)
            return None
        
        logger.debug(f"FPS: {fps}, Всего кадров: {total_frames}")

        frames = []

        duration_in_seconds = total_frames / fps  # Длительность видео в секундах
        start_frame = int(0.3 * fps)
        frame_interval = get_interval(duration_in_seconds)
        step = frame_interval * fps

        i = start_frame
        while total_frames > 0:
            last_flag = False
            # Проверка, что шаг последний и есть необходимость доп кадра
            remaining_frames = total_frames - i
            if step > remaining_frames > fps * 0.8:
                last_flag = True

            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, image = vidcap.read()
            if success:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                frames.append(pil_image)
            else:
                break

            if last_flag:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - start_frame)
                success, image = vidcap.read()
                if success:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    frames.append(pil_image)
                else:
                    break

            i = math.ceil(i + step)
            total_frames -= math.ceil(step)

        vidcap.release()
        logger.info(f"Успешно извлечено {len(frames)} кадров")
        return frames
    except Exception as e:
        error_msg = f"Ошибка нарезки ролика на кадры: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        return None


def get_interval(duration_in_seconds: Union[float, int]) -> Union[float, int]:
    if duration_in_seconds < 20:
        return 1
    elif duration_in_seconds < 31:
        return 1.3
    elif duration_in_seconds < 61:
        return 1.7
    else:
        return round(duration_in_seconds / 35, 1)


# -----------------------------
# Кодирование кадров
# -----------------------------
def encode_frames(frames: list[Union[str, Image.Image]]) -> list[str]:
    encoded_strings = []
    for frame in frames:
        encoded_string = encode_image(frame)
        encoded_strings.append(encoded_string)
    return encoded_strings


def encode_image(image: Image.Image, max_image=512) -> str:
    width, height = image.size
    max_dim = max(width, height)
    if max_dim > max_image:
        # Если изображение слишком большое, уменьшаем его размер
        scale_factor = max_image / max_dim
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height))

    # Сохраняем изображение в буфер и кодируем его в base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# -----------------------------
# Промпт для извлечения статистики Instagram
# -----------------------------
def get_instagram_stats_prompt() -> str:
    return """You are an expert in Instagram analytics focused on engagement and virality assessment. Your task: extract metrics from provided screenshots of Instagram Insights for one single post, then generate structured analysis and recommendations.

Follow ALL rules below strictly.

GENERAL RULES

Your input is screenshots of statistics for ONE post.

Extract ONLY information explicitly shown in images.

Never assume or infer numbers that are not clearly visible.

If a metric is missing → write "no data" (without quotes but as JSON string).

All numeric values must be numbers without text. Percentages as pure numbers.

Output must be valid JSON. No additional text before or after JSON.

If conflicting numbers appear in screenshots, use the most recent or detailed one.

After data extraction, perform engagement and virality assessment based solely on extracted data.

Always output recommendations if at least one engagement metric exists.

METRICS TO EXTRACT

post_url

post_type (photo / video / reels / carousel)

video_duration (seconds)

views_count

likes_count

shares_count (reposts)

comments_count

savings_count (saves)

views_breakdown:

followers_views

non_followers_views

followers_percentage

non_followers_percentage

average_watch_time:

seconds

percentage (relative to video duration)

total_watch_time

likes_to_views_ratio (in % or raw ratio)

traffic_sources:

profile

feed

reels_tab

stories

explore

other

AI ANALYSIS FIELDS

Always fill these if any engagement metric (views, likes, comments, shares, saves) is present.

engagement_assessment — concise evaluation of audience interaction quality (e.g. high like rate but poor retention).

completion_rate — estimated % of video completion if possible, else "no data".

virality_potential — direct estimate of potential reach (e.g. strong reactions but weak retention limits viral spread).

recommendations — list of concrete content improvements for reach, engagement, retention.

Analysis must be fact-based using extracted numbers. No motivation, no vague phrases.

OUTPUT FORMAT (strict)

{
  "post_url": "...",
  "post_type": "...",
  "video_duration": ...,
  "views_count": ...,
  "likes_count": ...,
  "shares_count": ...,
  "comments_count": ...,
  "savings_count": ...,
  "views_breakdown": {
    "followers_views": ...,
    "non_followers_views": ...,
    "followers_percentage": ...,
    "non_followers_percentage": ...
  },
  "average_watch_time": {
    "seconds": ...,
    "percentage": ...
  },
  "total_watch_time": ...,
  "likes_to_views_ratio": ...,
  "traffic_sources": {
    "profile": ...,
    "feed": ...,
    "reels_tab": ...,
    "stories": ...,
    "explore": ...,
    "other": ...
  },
  "ai_analysis": {
    "engagement_assessment": "...",
    "completion_rate": ...,
    "virality_potential": "...",
    "recommendations": "..."
  }
}

If values are missing, use:
"field_name": "no data"

No comments, no additional keys."""


def get_instagram_stats_prompt_multiple() -> str:
    """Prompt for extracting statistics from multiple screenshots of one post"""
    return """You are an expert in Instagram analytics focused on engagement and virality assessment. Your task: extract metrics from provided screenshots of Instagram Insights for one single post, then generate structured analysis and recommendations.

Follow ALL rules below strictly.

GENERAL RULES

Your input is SEVERAL SCREENSHOTS of statistics for the SAME post (different statistics tabs, different screens).

Extract ONLY information explicitly shown in images.

Never assume or infer numbers that are not clearly visible.

If a metric is missing → write "no data" (without quotes but as JSON string).

Combine data from all images into one consolidated result.

All numeric values must be numbers without text. Percentages as pure numbers.

Output must be valid JSON. No additional text before or after JSON.

If conflicting numbers appear in screenshots, use the most recent or detailed one.

After data extraction, perform engagement and virality assessment based solely on extracted data.

Always output recommendations if at least one engagement metric exists.

METRICS TO EXTRACT

post_url

post_type (photo / video / reels / carousel)

video_duration (seconds)

views_count

likes_count

shares_count (reposts)

comments_count

savings_count (saves)

views_breakdown:

followers_views

non_followers_views

followers_percentage

non_followers_percentage

average_watch_time:

seconds

percentage (relative to video duration)

total_watch_time

likes_to_views_ratio (in % or raw ratio)

traffic_sources:

profile

feed

reels_tab

stories

explore

other

AI ANALYSIS FIELDS

Always fill these if any engagement metric (views, likes, comments, shares, saves) is present.

engagement_assessment — concise evaluation of audience interaction quality (e.g. high like rate but poor retention).

completion_rate — estimated % of video completion if possible, else "no data".

virality_potential — direct estimate of potential reach (e.g. strong reactions but weak retention limits viral spread).

recommendations — list of concrete content improvements for reach, engagement, retention.

Analysis must be fact-based using extracted numbers. No motivation, no vague phrases.

OUTPUT FORMAT (strict)

{
  "post_url": "...",
  "post_type": "...",
  "video_duration": ...,
  "views_count": ...,
  "likes_count": ...,
  "shares_count": ...,
  "comments_count": ...,
  "savings_count": ...,
  "views_breakdown": {
    "followers_views": ...,
    "non_followers_views": ...,
    "followers_percentage": ...,
    "non_followers_percentage": ...
  },
  "average_watch_time": {
    "seconds": ...,
    "percentage": ...
  },
  "total_watch_time": ...,
  "likes_to_views_ratio": ...,
  "traffic_sources": {
    "profile": ...,
    "feed": ...,
    "reels_tab": ...,
    "stories": ...,
    "explore": ...,
    "other": ...
  },
  "ai_analysis": {
    "engagement_assessment": "...",
    "completion_rate": ...,
    "virality_potential": "...",
    "recommendations": "..."
  }
}

If values are missing, use:
"field_name": "no data"

No comments, no additional keys."""


# -----------------------------
# Анализ изображений с OpenAI
# -----------------------------
def analyze_images_with_openai(
    client: openai.OpenAI,
    encoded_strings: list[str],
    prompt: str,
    gpt_model: str = "gpt-4o",
):
    logger.info(f"Отправка запроса к {gpt_model} с {len(encoded_strings)} изображениями")
    
    # Начальная структура сообщения для OpenAI
    content_frames = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img}",
                "detail": "high",  # Используем high для лучшего распознавания текста
            },
        }
        for img in encoded_strings
    ]

    payload = {
        "model": gpt_model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert in Instagram analytics focused on engagement and virality assessment. Extract metrics accurately, perform fact-based analysis, and return only valid JSON.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *content_frames,
                ],
            },
        ],
        "temperature": 0,  # Низкая температура для более точного извлечения данных
        "response_format": {"type": "json_object"},  # Принудительно JSON формат
    }

    try:
        # Отправляем запрос в OpenAI для анализа изображений
        logger.debug("Отправка запроса к OpenAI API")
        response = client.chat.completions.create(**payload)
        logger.info("Получен ответ от OpenAI API")
        return response.choices[0].message.content
    except openai.APIError as e:
        error_msg = f"Ошибка API OpenAI: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise
    except Exception as e:
        error_msg = f"Неожиданная ошибка при запросе к OpenAI: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise


# -----------------------------
# Парсинг JSON ответа
# -----------------------------
def parse_json_response(response: str) -> dict:
    """Парсит JSON из ответа GPT и возвращает словарь с дефолтными значениями."""
    
    # Пытаемся найти JSON в ответе (на случай, если есть лишний текст)
    response = response.strip()
    
    # Если ответ начинается с ```json или ```, удаляем это
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    
    # Если ответ заканчивается на ```, удаляем это
    if response.endswith("```"):
        response = response[:-3]
    
    response = response.strip()
    
    # Парсим JSON
    try:
        data = json.loads(response)
        logger.debug("JSON успешно распарсен")
    except json.JSONDecodeError as e:
        error_msg = f"Невалидный JSON в ответе: {e}\nОтвет: {response}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Create structure with default values
    default_structure = {
        "post_url": "no data",
        "post_type": "no data",
        "video_duration": "no data",
        "views_count": "no data",
        "likes_count": "no data",
        "shares_count": "no data",
        "comments_count": "no data",
        "savings_count": "no data",
        "views_breakdown": {
            "followers_views": "no data",
            "non_followers_views": "no data",
            "followers_percentage": "no data",
            "non_followers_percentage": "no data",
        },
        "average_watch_time": {
            "seconds": "no data",
            "percentage": "no data",
        },
        "total_watch_time": "no data",
        "likes_to_views_ratio": "no data",
        "traffic_sources": {
            "profile": "no data",
            "feed": "no data",
            "reels_tab": "no data",
            "stories": "no data",
            "explore": "no data",
            "other": "no data",
        },
        "ai_analysis": {
            "engagement_assessment": "no data",
            "completion_rate": "no data",
            "virality_potential": "no data",
            "recommendations": "no data",
        },
    }
    
    # Merge received data with default structure
    result = _deep_merge(default_structure, data)
    
    # If we have sufficient data, ensure AI analysis is provided
    # Check if any engagement metric exists (views, likes, comments, shares, saves)
    has_sufficient_data = (
        result.get("views_count") != "no data" or
        result.get("likes_count") != "no data" or
        result.get("comments_count") != "no data" or
        result.get("shares_count") != "no data" or
        result.get("savings_count") != "no data"
    )
    
    if has_sufficient_data:
        # Ensure AI analysis fields are filled if they are "no data"
        ai_analysis = result.get("ai_analysis", {})
        if ai_analysis.get("engagement_assessment") == "no data":
            ai_analysis["engagement_assessment"] = "Analysis will be provided based on available metrics"
        if ai_analysis.get("virality_potential") == "no data":
            ai_analysis["virality_potential"] = "Analysis will be provided based on available metrics"
        if ai_analysis.get("recommendations") == "no data":
            ai_analysis["recommendations"] = "Recommendations will be provided based on available metrics"
    
    return result


def _deep_merge(base: dict, update: dict) -> dict:
    """Рекурсивно мержит два словаря."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# -----------------------------
# Пример использования
# -----------------------------
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python instagram_stats_extractor.py <PATH_TO_FILE> [API_KEY]")
        print("Пример: python instagram_stats_extractor.py video.mp4")
        print("  или: python instagram_stats_extractor.py video.mp4 sk-...")
        print("\nПримечание: API ключ можно также установить через переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    file_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        stats = extract_instagram_stats(
            file_path=file_path,
            api_key=api_key,
        )
        
        # Выводим результат в формате JSON
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

