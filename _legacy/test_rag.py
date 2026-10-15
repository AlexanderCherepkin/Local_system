"""Тестирование RAG системы."""

import logging
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def test_ollama_connection():
    """Проверить подключение к Ollama."""
    print("\n=== Проверка подключения к Ollama ===")
    
    import requests
    
    base_url = "http://localhost:11434"
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            
            print(f"✅ Ollama доступен")
            print(f"   Доступные модели: {', '.join(models) if models else 'Нет моделей'}")
            
            return True, models
        else:
            print(f"❌ Ollama вернул код статуса: {response.status_code}")
            return False, []
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Cannot connect to Ollama at {base_url}")
        print("   Запустите: ollama serve")
        print("   Или убедитесь, что localhost:11434 доступен")
        return False, []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, []


def test_generator():
    """Тестирование генератора."""
    print("\n=== Тестирование Generator ===")
    
    try:
        from pipeline.generator import OllamaGenerator
        
        generator = OllamaGenerator()
        
        # Проверка параметров
        print(f"✅ Model name: {generator.model_name}")
        print(f"   Top-K: {generator.top_k}")
        print(f"   Temperature: {generator.temperature}")
        print(f"   Similarity threshold: {generator.similarity_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing generator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_pipeline():
    """Тестирование RAG пайплайна."""
    print("\n=== Тестирование RAG Pipeline ===")
    
    try:
        from pipeline.generator import RAGPipeline
        
        pipeline = RAGPipeline()
        
        # Проверка настроек
        if hasattr(pipeline, 'settings'):
            print(f"✅ Settings loaded successfully")
            print(f"   DATA_DIR: {pipeline.settings.DATA_DIR}")
            print(f"   CHUNK_SIZE: {pipeline.settings.CHUNK_SIZE}")
            print(f"   TOP_K_RESULTS: {pipeline.settings.TOP_K_RESULTS}")
        else:
            print("⚠ Warning: Settings not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запуск всех тестов."""
    
    print("=" * 60)
    print("RAG System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Ollama Connection", test_ollama_connection),
        ("Generator Module", test_generator),
        ("RAG Pipeline", test_rag_pipeline)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Итоговый отчет
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(s for _, s in results)


if __name__ == "__main__":
    main()
