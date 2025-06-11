Инструкция по запуску
Установка виртуального окружения
python -m venv myenv

Активация виртуального окружения
myenv\Scripts\activate

Установка зависимостей
pip install -r requirements.txt

Добавьте путь к Tesseract в PATH
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"

Запускаем программу
python knowledge_base.py

Останавливаем программу