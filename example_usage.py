"""
Пример использования Instagram Stats Extractor
"""
import json
import os
from instagram_stats_extractor import extract_instagram_stats

# Укажите путь к файлу (видео или скриншот)
FILE_PATH = "IMG_8120.png"  # или "video.mp4"

if __name__ == "__main__":
    # Проверяем наличие API ключа в переменных окружения
    if not os.getenv("OPENAI_API_KEY"):
        print("ВНИМАНИЕ: Переменная окружения OPENAI_API_KEY не установлена!")
        print("Установите её командой:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  или в Windows:")
        print("  set OPENAI_API_KEY=sk-...")
        print("\nПродолжаем с попыткой использовать ключ из переменных окружения...\n")
    
    if not os.path.exists(FILE_PATH):
        print(f"Файл не найден: {FILE_PATH}")
        print("Доступные файлы:")
        for f in os.listdir("."):
            if os.path.isfile(f) and not f.startswith("."):
                print(f"  - {f}")
    else:
        try:
            print(f"Анализ файла: {FILE_PATH}")
            print("Ожидайте, это может занять некоторое время...\n")
            
            # API ключ будет автоматически взят из переменной окружения OPENAI_API_KEY
            stats = extract_instagram_stats(
                file_path=FILE_PATH,
            )
            
            # Выводим результат в читаемом формате
            print("=" * 60)
            print("ИЗВЛЕЧЕННАЯ СТАТИСТИКА")
            print("=" * 60)
            print(json.dumps(stats, ensure_ascii=False, indent=2))
            
            # Сохраняем в файл
            output_file = f"stats_{os.path.splitext(FILE_PATH)[0]}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"\nРезультат сохранен в: {output_file}")
            
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()

