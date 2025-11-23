# Instagram Stats Extractor

Инструмент для извлечения статистики постов Instagram из видео записей или скриншотов с использованием ChatGPT 4o.

## Установка

```bash
pip install -r requirements.txt
```

## Настройка API ключа

Рекомендуемый способ — использовать переменную окружения:

**Linux/macOS:**
```bash
export OPENAI_API_KEY='sk-...'
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=sk-...
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY='sk-...'
```

## Использование

### Графический интерфейс (GUI) - Рекомендуется

Самый простой способ использования — через графический интерфейс:

```bash
python instagram_gui.py
```

**Возможности GUI:**
- Выбор до 5 файлов одновременно
- Удобный просмотр результатов в отдельных вкладках
- Сохранение результатов в JSON
- Копирование результатов в буфер обмена
- Прогресс-бар и статус обработки
- Автоматическое форматирование результатов

**Инструкция:**
1. Запустите `python instagram_gui.py`
2. Нажмите "Выбрать файлы" и выберите до 5 файлов
3. Нажмите "Запустить анализ"
4. Дождитесь завершения анализа
5. Просмотрите результаты во вкладках
6. При необходимости сохраните результаты через контекстное меню (правый клик)

### Из командной строки

```bash
# С использованием переменной окружения (рекомендуется)
python instagram_stats_extractor.py <PATH_TO_FILE>

# Или с явным указанием API ключа
python instagram_stats_extractor.py <PATH_TO_FILE> <API_KEY>
```

Примеры:
```bash
# Для видео (используя переменную окружения)
python instagram_stats_extractor.py video.mp4

# Для скриншота (с явным указанием ключа)
python instagram_stats_extractor.py screenshot.png sk-...
```

### Из Python кода

```python
from instagram_stats_extractor import extract_instagram_stats

# API ключ автоматически берется из переменной окружения OPENAI_API_KEY
stats = extract_instagram_stats(
    file_path="video.mp4",  # или "screenshot.png"
)

# Или можно указать ключ явно
stats = extract_instagram_stats(
    file_path="video.mp4",
    api_key="sk-...",  # опционально
)

print(stats)
```

## Формат выходных данных

Инструмент возвращает JSON со следующей структурой:

```json
{
  "post_url": "ссылка на пост или no data",
  "post_type": "тип поста (photo/video/reels/carousel) или no data",
  "video_duration": "длительность видео в секундах или no data",
  "views_count": "количество просмотров или no data",
  "shares_count": "количество репостов или no data",
  "comments_count": "количество комментариев или no data",
  "savings_count": "количество сохранений или no data",
  "views_breakdown": {
    "followers_views": "просмотры от подписчиков или no data",
    "non_followers_views": "просмотры от не подписчиков или no data",
    "followers_percentage": "доля просмотров от подписчиков в % или no data",
    "non_followers_percentage": "доля просмотров от не подписчиков в % или no data"
  },
  "average_watch_time": {
    "seconds": "средняя длительность просмотра в секундах или no data",
    "percentage": "средняя длительность просмотра в % или no data"
  },
  "total_watch_time": "общее время просмотра (в секундах или часах) или no data",
  "likes_to_views_ratio": "соотношение лайков к просмотрам или no data",
  "traffic_sources": {
    "home": "просмотры из ленты или no data",
    "profile": "просмотры с профиля или no data",
    "hashtags": "просмотры по хештегам или no data",
    "explore": "просмотры из раздела Explore или no data",
    "other": "другие источники или no data"
  },
  "ai_analysis": {
    "engagement_assessment": "мнение нейросети про вовлеченность или no data",
    "completion_rate": "оценка % досмотров или no data",
    "virality_potential": "оценка потенциала виральности или no data"
  },
  "recommendations": "выводы и рекомендации нейросети или no data"
}
```

## Поддерживаемые форматы

- **Видео**: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm, .m4v
- **Изображения**: .png, .jpg, .jpeg, .bmp, .gif и другие форматы, поддерживаемые PIL

## Особенности

- Автоматическое определение типа файла (видео или изображение)
- Для видео: автоматическая нарезка на ключевые кадры
- Для изображений: прямое использование скриншота
- Точное извлечение только видимых данных (без выдумывания)
- Если данные отсутствуют, используется значение "no data"
- Анализ вовлеченности и рекомендации от GPT-4o
- **Логирование всех операций и ошибок** в файлы:
  - `instagram_extractor.log` - логи основного модуля
  - `instagram_gui.log` - логи GUI приложения

## Требования

- Python 3.8+
- OpenAI API ключ с доступом к GPT-4o
- Установленные зависимости из requirements.txt

