# Руководство по добавлению новой модели в систему

Это руководство описывает процесс добавления новой модели в систему с использованием классов ChatModel и ModelCreator. Следуя этим шагам, вы сможете интегрировать свою модель в существующую архитектуру чатов.

## Содержание

1. Обзор архитектуры
2. Создание класса модели
3. Регистрация модели в ModelCreator
4. Тестирование новой модели
5. Примеры


## Обзор архитектуры

Система использует следующие ключевые компоненты:
- `ChatModel` - базовый интерфейс для всех моделей чата
- `ModelCreator` - фабрика для создания экземпляров моделей
- `ChatManager` - управляет чатами и их взаимодействием с моделями

`ModelCreator` для создания моделей вызывает функции, и передает им ссылку на llm и ссылку на словарь с настройками приложения (по умолчанию заполняется из переменных среды).

## Создание класса модели

### Шаг 1: Создайте новый класс модели

Создайте новый файл Python для вашей модели в соответствующем каталоге (например, simple_rag/models/your_model/model.py):

```python
from typing import Dict, Any, Optional

from langchain.chat_models.base import BaseChatModel
from simple_rag.models.base import ChatModel

class YourCustomModel(ChatModel):
    """
    Ваша пользовательская модель чата.
    """
    
    def __init__(self, 
        llm: BaseChatModel, 
        config: Dict[str, Any]
    ):
        """
        Инициализация модели с конфигурацией.
        
        Args:
            config: Словарь с параметрами конфигурации. 
        """
        self.config = config
        # Инициализируйте здесь все необходимые компоненты
        # Например:
        self.llm = llm
        self.store = get_store(config['supermodel_store_name'])
        # Другие параметры...
    
    def send(self, message: str) -> str:
        """
        Генерирует ответ на сообщение пользователя.
        
        Args:
            message: Сообщение пользователя
            
        Returns:
            Ответ модели
        """
        # Реализуйте логику генерации ответа
        # Например:
        # response = your_model_logic(message, history, self.config)
        # return response
        
        # Временная заглушка:
        return f"Ответ от {self.model_name} на сообщение: {message}"
    
    def update(self, config_updates: Dict[str, Any]) -> None:
        """
        Обновляет конфигурацию модели.
        
        Args:
            config_updates: Словарь с обновлениями конфигурации
        """
        self.config.update(config_updates)
        # Обновите внутренние параметры в соответствии с новой конфигурацией
        if "model_name" in config_updates:
            self.model_name = config_updates["model_name"]
        if "temperature" in config_updates:
            self.temperature = config_updates["temperature"]
        # Другие обновления...
```

### Шаг 2: Создайте функцию-строитель для вашей модели
Создайте функцию, которая будет строить экземпляр вашей модели:

```python
def build_your_custom_model(llm, config: Dict[str, Any]) -> YourCustomModel:
    """
    Создает экземпляр YourCustomModel с заданной конфигурацией.
    
    Args:
        llm: Экземпляр llm
        config: Словарь с параметрами конфигурации
        
    Returns:
        Экземпляр YourCustomModel
    """
    # Здесь можно выполнить дополнительную валидацию или преобразование конфигурации
    return YourCustomModel(llm, config)
```

## Регистрация модели в ModelCreator

### Шаг 3: Импортируйте вашу модель в файл инициализации

Добавьте импорт вашей функции-строителя в соответствующий файл инициализации (`simple_rag/models/__init__.py`):

```python
from simple_rag.models.your_model.model import build_your_custom_model
```

### Шаг 4: "Зарегистрируйте" вашу модель в ModelCreator

Найдите код, где инициализируется ModelCreator, и добавьте вашу модель:

В файле, где инициализируется ModelCreator (например, simple_rag/models/__init__.py)

```python
class ModelCreator:

    ...

    # FIXME: config should be generalized or the whole approach to models should be changed
    def __init__(self, llm: BaseChatModel, config: QnAServiceConfig):
        ...

        self._models = {
            ...
            'your_model': build_your_custom_model
        }
```

## Тестирование новой модели

### Шаг 5: Создайте тесты для вашей модели

Создайте файл с тестами для вашей модели (например, tests/test_your_custom_model.py):

```python
import pytest
from unittest.mock import MagicMock
from simple_rag.models.your_model.model import YourCustomModel, build_your_custom_model

def test_your_custom_model_initialization():
    """Тест инициализации модели с конфигурацией."""
    llm = MagicMock()
    config = {
        "model_name": "test_model",
        "temperature": 0.5
    }
    model = YourCustomModel(llm, config)
    
    assert model.model_name == "test_model"
    assert model.temperature == 0.5

def test_your_custom_model_generate_response():
    """Тест генерации ответа."""
    llm = MagicMock()
    config = {"model_name": "test_model"}
    model = YourCustomModel(llm, config)
    
    response = model.send("Привет, как дела?")
    assert isinstance(response, str)
    assert len(response) > 0

def test_your_custom_model_update():
    """Тест обновления конфигурации модели."""
    config = {
        "model_name": "test_model",
        "temperature": 0.5
    }
    llm = MagicMock()
    model = YourCustomModel(llm, config)
    
    model.update({"temperature": 0.8, "model_name": "updated_model"})
    
    assert model.temperature == 0.8
    assert model.model_name == "updated_model"

def test_build_your_custom_model():
    """Тест функции-строителя."""
    llm = MagicMock()
    config = {"model_name": "test_model"}
    model = build_your_custom_model(llm, config)
    
    assert isinstance(model, YourCustomModel)
    assert model.model_name == "test_model"

```

### Шаг 6: Проверьте интеграцию с ChatManager

Создайте тест для проверки интеграции вашей модели с `ChatManager`:


```python
def test_chat_manager_with_your_custom_model():
    """Тест интеграции с ChatManager."""
    from simple_rag.chats import ChatManager
    from simple_rag.models.your_model.model import build_your_custom_model
    
    config = {"model_name": "test_model"}
    model = build_your_custom_model(config)
    
    chat_manager = ChatManager()
    chat = chat_manager.create_chat(model)
    
    response = chat_manager.send_message(chat.id, "Тестовое сообщение")
    
    assert isinstance(response, str)
    assert len(response) > 0
```

## Примеры

### Пример использования через API

После регистрации вашей модели в ModelCreator, вы можете использовать её через API:

```bash
# Создание чата с вашей моделью
curl -X POST "http://localhost:8000/chat/create" \
     -H "Content-Type: application/json" \
     -d '{"model": "your_custom_model"}'

# Ответ:
# {"chat_id": "550e8400-e29b-41d4-a716-446655440000"}

# Отправка сообщения
curl -X POST "http://localhost:8000/chat/message" \
     -H "Content-Type: application/json" \
     -d '{"chat_id": "550e8400-e29b-41d4-a716-446655440000", "message": "Привет, как дела?"}'

# Ответ:
# {"response": "Ответ от custom_model_v1 на сообщение: Привет, как дела?"}

# Обновление параметров модели
curl -X POST "http://localhost:8000/chat/update_model" \
     -H "Content-Type: application/json" \
     -d '{"chat_id": "550e8400-e29b-41d4-a716-446655440000", "prompt": "Новый промпт"}'

# Удаление чата
curl -X DELETE "http://localhost:8000/chat/550e8400-e29b-41d4-a716-446655440000"
```

##  Заключение

Следуя этому руководству, вы можете добавить новую модель в систему, которая будет полностью интегрирована с существующей архитектурой чатов. Убедитесь, что ваша модель правильно реализует интерфейс ChatModel и корректно обрабатывает все необходимые методы.
Помните о необходимости тщательного тестирования вашей модели перед использованием в производственной среде.