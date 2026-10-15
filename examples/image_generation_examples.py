"""
Примеры использования сервиса генерации изображений.
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.image_generation import (
    detect_image_generation_intent,
    generate_image,
    generate_image_variations
)
from utils.logging import logger


async def example_1_detect_intent():
    """
    Пример 1: Определение намерения генерации изображения.
    """
    print("\n" + "="*60)
    print("Пример 1: Определение намерения")
    print("="*60)
    
    test_phrases = [
        "Нарисуй кота в космосе",
        "Что такое Python?",
        "Создай изображение заката",
        "Как дела?",
        "Сгенерируй картинку дракона",
        "Расскажи про машинное обучение"
    ]
    
    for phrase in test_phrases:
        result = await detect_image_generation_intent(phrase)
        print(f"\nФраза: '{phrase}'")
        print(f"  Нужна генерация: {result['needs_generation']}")
        print(f"  Уверенность: {result.get('confidence', 0):.2f}")
        if result.get('prompt'):
            print(f"  Промпт: {result['prompt'][:80]}...")


async def example_2_generate_simple():
    """
    Пример 2: Простая генерация изображения.
    """
    print("\n" + "="*60)
    print("Пример 2: Простая генерация")
    print("="*60)
    
    prompt = "A majestic cat in a space suit floating in space, stars in the background"
    
    print(f"\nГенерация изображения...")
    print(f"Промпт: {prompt}")
    
    try:
        result = await generate_image(prompt)
        
        print(f"\n✅ Изображение создано!")
        print(f"  Путь: {result['image_path']}")
        print(f"  Улучшенный промпт: {result['revised_prompt'][:100]}...")
        print(f"  URL: {result['url']}")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


async def example_3_generate_with_params():
    """
    Пример 3: Генерация с кастомными параметрами.
    """
    print("\n" + "="*60)
    print("Пример 3: Генерация с параметрами")
    print("="*60)
    
    prompt = "A futuristic city with flying cars and neon signs, cyberpunk style"
    
    print(f"\nГенерация изображения с параметрами...")
    print(f"Промпт: {prompt}")
    print(f"Размер: 1792x1024 (горизонтальный)")
    print(f"Качество: hd")
    print(f"Стиль: vivid")
    
    try:
        result = await generate_image(
            prompt=prompt,
            size="1792x1024",  # Горизонтальное изображение
            quality="hd",  # Высокое качество
            style="vivid"  # Яркий стиль
        )
        
        print(f"\n✅ Изображение создано!")
        print(f"  Путь: {result['image_path']}")
        print(f"  Улучшенный промпт: {result['revised_prompt'][:100]}...")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


async def example_4_multiple_generations():
    """
    Пример 4: Генерация нескольких изображений по очереди.
    """
    print("\n" + "="*60)
    print("Пример 4: Множественная генерация")
    print("="*60)
    
    prompts = [
        "A red apple on a wooden table",
        "A blue butterfly on a flower",
        "A golden sunset over mountains"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Генерация: {prompt}")
        
        try:
            result = await generate_image(
                prompt=prompt,
                size="1024x1024",
                quality="standard"
            )
            print(f"  ✅ Создано: {result['image_path']}")
            
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
        
        # Небольшая задержка между запросами
        if i < len(prompts):
            await asyncio.sleep(2)


async def example_5_variations():
    """
    Пример 5: Генерация вариаций существующего изображения.
    
    Примечание: Требуется существующее изображение PNG с прозрачным фоном.
    """
    print("\n" + "="*60)
    print("Пример 5: Генерация вариаций")
    print("="*60)
    
    # Сначала создаем базовое изображение
    print("\nШаг 1: Создание базового изображения...")
    
    try:
        base_result = await generate_image(
            "A simple red circle on transparent background",
            size="1024x1024"
        )
        
        print(f"  ✅ Базовое изображение: {base_result['image_path']}")
        
        # Генерируем вариации
        print("\nШаг 2: Генерация вариаций...")
        
        # Примечание: API требует PNG с прозрачным фоном
        # В реальном использовании нужно подготовить изображение
        print("  ℹ️ Для генерации вариаций нужно PNG с прозрачным фоном")
        print("  ℹ️ Пропускаем этот пример...")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


async def example_6_with_conversation_context():
    """
    Пример 6: Определение намерения с контекстом разговора.
    """
    print("\n" + "="*60)
    print("Пример 6: Определение с контекстом")
    print("="*60)
    
    # Симулируем историю разговора
    conversation_history = [
        {"role": "user", "content": "Привет! Как дела?"},
        {"role": "assistant", "content": "Привет! Все отлично, чем могу помочь?"},
        {"role": "user", "content": "Расскажи про космос"},
        {"role": "assistant", "content": "Космос - это..."}
    ]
    
    current_message = "А можешь показать как это выглядит?"
    
    print(f"\nИстория разговора:")
    for msg in conversation_history[-2:]:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    print(f"\nТекущее сообщение: '{current_message}'")
    
    result = await detect_image_generation_intent(
        current_message,
        conversation_history
    )
    
    print(f"\nРезультат:")
    print(f"  Нужна генерация: {result['needs_generation']}")
    print(f"  Уверенность: {result.get('confidence', 0):.2f}")
    
    if result.get('needs_generation'):
        print(f"  Контекстный промпт: {result.get('prompt', '')[:80]}...")


async def example_7_error_handling():
    """
    Пример 7: Обработка ошибок.
    """
    print("\n" + "="*60)
    print("Пример 7: Обработка ошибок")
    print("="*60)
    
    # Пример с потенциально проблемным контентом
    problematic_prompts = [
        "A violent scene",  # Может нарушать политику
        "",  # Пустой промпт
    ]
    
    for prompt in problematic_prompts:
        print(f"\nТест: '{prompt}'")
        
        try:
            result = await generate_image(prompt)
            print(f"  ✅ Успешно: {result['image_path']}")
            
        except Exception as e:
            print(f"  ❌ Ожидаемая ошибка: {type(e).__name__}")
            print(f"     Сообщение: {str(e)[:100]}")


async def main():
    """
    Главная функция для запуска примеров.
    """
    print("\n" + "="*60)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ГЕНЕРАЦИИ ИЗОБРАЖЕНИЙ")
    print("="*60)
    
    examples = {
        "1": ("Определение намерения", example_1_detect_intent),
        "2": ("Простая генерация", example_2_generate_simple),
        "3": ("Генерация с параметрами", example_3_generate_with_params),
        "4": ("Множественная генерация", example_4_multiple_generations),
        "5": ("Генерация вариаций", example_5_variations),
        "6": ("Определение с контекстом", example_6_with_conversation_context),
        "7": ("Обработка ошибок", example_7_error_handling),
    }
    
    print("\nДоступные примеры:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Запустить все примеры")
    print("  q. Выход")
    
    choice = input("\nВыберите пример (или 'q' для выхода): ").strip()
    
    if choice == 'q':
        print("Выход...")
        return
    
    if choice == '0':
        print("\nЗапуск всех примеров...\n")
        for name, func in examples.values():
            await func()
            await asyncio.sleep(1)
    elif choice in examples:
        name, func = examples[choice]
        print(f"\nЗапуск: {name}\n")
        await func()
    else:
        print("❌ Неверный выбор")
    
    print("\n" + "="*60)
    print("Примеры завершены!")
    print("="*60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
    except Exception as e:
        logger.error(f"Ошибка в примерах: {e}", exc_info=True)
        print(f"\n❌ Ошибка: {e}")

