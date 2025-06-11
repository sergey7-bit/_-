import os
import re
import json
import logging
import requests
import fitz
import io
import pytesseract
import numpy as np
import concurrent.futures
from bs4 import BeautifulSoup
from docx import Document
from PIL import Image
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import jensenshannon

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parsing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Конфигурация
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
TIMEOUT = 50 
MAX_WORKERS = 5
OUTPUT_DIR = "parsed_documents"
METRICS_FILE = "parsing_metrics.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class DocumentParser:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.processed_urls = set()

    def download_document(self, url: str) -> Optional[Dict]:
        """Загрузка документа с обработкой ошибок"""
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            extension = urlparse(url).path.split('.')[-1].lower() if '.' in urlparse(url).path else ''
            
            doc_type = self._determine_doc_type(content_type, extension)
            if doc_type == 'unknown':
                logger.warning(f"Неизвестный тип документа: {url}")
                return None
                
            return {
                'content': response.content,
                'type': doc_type,
                'url': url,
                'last_modified': response.headers.get('Last-Modified', '')
            }
        except Exception as e:
            logger.error(f"Ошибка загрузки {url}: {str(e)}")
            return None

    def _determine_doc_type(self, content_type: str, extension: str) -> str:
        """Определение типа документа"""
        if 'text/html' in content_type or extension in ('html', 'htm'):
            return 'html'
        elif 'application/pdf' in content_type or extension == 'pdf':
            return 'pdf'
        elif 'vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type or extension == 'docx':
            return 'docx'
        elif 'text/plain' in content_type or extension in ('txt', 'md'):
            return 'text'
        return 'unknown'

    def parse_document(self, url: str) -> Optional[Dict]:
        """Основной метод парсинга документа"""
        if url in self.processed_urls:
            return None
            
        doc = self.download_document(url)
        if not doc:
            return None
            
        try:
            parser = getattr(self, f"_parse_{doc['type']}", None)
            if not parser:
                logger.warning(f"Нет парсера для типа {doc['type']}: {url}")
                return None
                
            content, metadata = parser(doc['content'], url)
            if not content.strip():
                logger.warning(f"Пустое содержимое: {url}")
                return None
                
            # Оценка качества
            quality_metrics = self._assess_quality(content, url)
            
            # Сохранение документа
            filename = self._save_document(content, metadata)
            
            result = {
                'url': url,
                'type': doc['type'],
                'filename': filename,
                'metadata': metadata,
                'metrics': quality_metrics
            }
            
            self.processed_urls.add(url)
            return result
            
        except Exception as e:
            logger.error(f"Ошибка парсинга {url}: {str(e)}", exc_info=True)
            return None

    def _parse_html(self, content: bytes, url: str) -> Tuple[str, Dict]:
        """Парсинг HTML"""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Удаление ненужных элементов
        for element in soup(['script', 'style', 'header', 'footer', 'aside', 'nav']):
            element.decompose()
            
        # Извлечение текста с сохранением структуры
        text = ""
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'table']):
            if element.name.startswith('h'):
                level = int(element.name[1])
                text += f"{'#' * level} {element.get_text()}\n\n"
            elif element.name == 'p':
                text += f"{element.get_text()}\n\n"
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    prefix = "- " if element.name == 'ul' else "1. "
                    text += f"{prefix}{li.get_text()}\n"
                text += "\n"
            elif element.name == 'table':
                text += self._parse_table(element)
                
        metadata = {
            'title': soup.title.string if soup.title else urlparse(url).path,
            'source': url,
            'type': 'html',
            'processed_at': datetime.now().isoformat()
        }
        
        return text, metadata

    def _parse_pdf(self, content: bytes, url: str) -> Tuple[str, Dict]:
        """Парсинг PDF с поддержкой OCR"""
        text = ""
        metadata = {
            'source': url,
            'type': 'pdf',
            'processed_at': datetime.now().isoformat()
        }
        
        with fitz.open(stream=content, filetype="pdf") as pdf:
            metadata['pages'] = len(pdf)
            metadata['title'] = pdf.metadata.get('title', '') or os.path.basename(urlparse(url).path)
            
            for page_num, page in enumerate(pdf):
                page_text = page.get_text()
                
                # OCR для сканированных документов
                if not page_text.strip():
                    try:
                        pix = page.get_pixmap(dpi=200)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        page_text = pytesseract.image_to_string(img, lang='rus+eng')
                        logger.info(f"Применен OCR для страницы {page_num+1} в {url}")
                    except Exception as e:
                        logger.warning(f"Ошибка OCR для {url} страница {page_num+1}: {str(e)}")
                
                text += f"\n\n## Страница {page_num+1}\n\n{page_text}"
        
        return text, metadata

    def _parse_docx(self, content: bytes, url: str) -> Tuple[str, Dict]:
        """Парсинг DOCX"""
        docx = Document(io.BytesIO(content))
        text = ""
        
        # Обработка параграфов
        for para in docx.paragraphs:
            if not para.text.strip():
                continue
                
            if para.style.name.startswith('Heading'):
                level = int(para.style.name.split(' ')[1])
                text += f"{'#' * level} {para.text}\n\n"
            else:
                text += f"{para.text}\n\n"
        
        # Обработка таблиц
        for table in docx.tables:
            text += "\n\n**Таблица:**\n\n"
            for i, row in enumerate(table.rows):
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if not cells:
                    continue
                    
                if i == 0:  # Заголовок таблицы
                    text += "| " + " | ".join(cells) + " |\n"
                    text += "| " + " | ".join(["---"] * len(cells)) + " |\n"
                else:  # Данные таблицы
                    text += "| " + " | ".join(cells) + " |\n"
            text += "\n"
        
        metadata = {
            'source': url,
            'type': 'docx',
            'title': docx.core_properties.title or os.path.basename(urlparse(url).path),
            'author': docx.core_properties.author,
            'created': str(docx.core_properties.created),
            'modified': str(docx.core_properties.modified),
            'processed_at': datetime.now().isoformat()
        }
        
        return text, metadata

    def _parse_text(self, content: bytes, url: str) -> Tuple[str, Dict]:
        """Парсинг текстовых файлов"""
        text = content.decode('utf-8', errors='replace')
        metadata = {
            'source': url,
            'type': 'text',
            'title': os.path.basename(urlparse(url).path),
            'processed_at': datetime.now().isoformat()
        }
        return text, metadata

    def _parse_table(self, table) -> str:
        """Парсинг HTML таблицы в Markdown"""
        headers = [th.get_text().strip() for th in table.find_all('th')]
        rows = []
        
        for tr in table.find_all('tr'):
            row = [td.get_text().strip() for td in tr.find_all('td')]
            if row:
                rows.append(row)
                
        if not headers and rows:
            headers = [f"Column {i+1}" for i in range(len(rows[0]))]
            
        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        for row in rows:
            table_md += "| " + " | ".join(row) + " |\n"
            
        return table_md + "\n"

    def _save_document(self, content: str, metadata: Dict) -> str:
        """Сохранение документа в файл"""
        title = re.sub(r'[^\w\-_\. ]', '_', metadata.get('title', 'document'))[:50]
        source = urlparse(metadata['source']).path.replace('/', '_')[:50]
        filename = f"{title}_{source}.md"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return filename

    def _assess_quality(self, content: str, url: str) -> Dict:
        """Оценка качества парсинга с использованием различных метрик"""
        metrics = {}
        
        # Оценка структуры
        metrics['structure_score'] = self._calculate_structure_score(content)
        
        # Оценка уникальности (упрощенная версия)
        metrics['uniqueness'] = self._calculate_uniqueness(content)
        
        # Сохранение метрик
        self.metrics[url] = metrics
        
        return metrics

    def _calculate_structure_score(self, text: str) -> float:
        """Оценка структурированности текста"""
        lines = text.split('\n')
        headings = sum(1 for line in lines if line.startswith('#'))
        lists = sum(1 for line in lines if line.startswith(('-', '*', '1.')))
        tables = sum(1 for line in lines if '|' in line and '---' in line)
        
        total_elements = headings + lists + tables
        total_lines = len(lines)
        
        if total_lines == 0:
            return 0.0
            
        return min(1.0, total_elements / (total_lines / 10))

    def _calculate_uniqueness(self, text: str) -> float:
        """Упрощенная оценка уникальности текста"""
        words = re.findall(r'\w+', text.lower())
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0.0

    def generate_report(self) -> Dict:
        """Генерация отчета о парсинге"""
        report = {
            'summary': {
                'total_urls': len(self.metrics),
                'successful_parses': len([m for m in self.metrics.values() if m]),
                'success_rate': len([m for m in self.metrics.values() if m]) / len(self.metrics) if self.metrics else 0,
                'document_types': defaultdict(int),
                'avg_structure_score': 0,
                'avg_uniqueness': 0,
            },
            'details': []
        }
        
        # Анализ по типам документов
        type_counts = defaultdict(int)
        structure_scores = []
        uniqueness_scores = []
        
        for url, metrics in self.metrics.items():
            doc_type = next((d['type'] for d in self.processed_urls if d['url'] == url), 'unknown')
            type_counts[doc_type] += 1
            
            if metrics:
                structure_scores.append(metrics.get('structure_score', 0))
                uniqueness_scores.append(metrics.get('uniqueness', 0))
                report['details'].append({
                    'url': url,
                    'type': doc_type,
                    'metrics': metrics
                })
        
        report['summary']['document_types'] = dict(type_counts)
        report['summary']['avg_structure_score'] = sum(structure_scores) / len(structure_scores) if structure_scores else 0
        report['summary']['avg_uniqueness'] = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0
        
        # Сохранение отчета
        with open(os.path.join(OUTPUT_DIR, METRICS_FILE), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        return report

def main(urls: List[str]):
    """Основная функция для выполнения парсинга"""
    parser = DocumentParser()
    
    # Многопоточный парсинг
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(parser.parse_document, url): url for url in urls}
        
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                result = future.result()
                if result:
                    logger.info(f"Успешно обработан: {url}")
            except Exception as e:
                logger.error(f"Ошибка при обработке {url}: {str(e)}")
    
    # Генерация отчета
    report = parser.generate_report()
    logger.info(f"\nОтчет о парсинге:\n{json.dumps(report['summary'], indent=2)}")
    
    # Сохранение полного отчета
    with open('parsing_summary.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Полный отчет сохранен в parsing_summary.json")

if __name__ == "__main__":
    urls = [
        "https://minzdrav.gov.ru/documents/7025-federalnyy-zakon-323-fz-ot-21-noyabrya-2011-g",
        "https://www.garant.ru/products/ipo/prime/doc/405742919/",
        "https://base.garant.ru/71431038/",
        "https://docs.cntd.ru/document/542615520",
        "https://base.garant.ru/72147566/#:~:text=декабря%202018%20г.-,N%20898н%20%22О%20внесении%20изменений%20в%20сроки%20и%20этапы%20аккредитации,от%2022%20декабря%202017%20г.",
        "https://minzdrav.gov.ru/documents/8956-prikaz-ministerstva-zdravoohraneniya-rossiyskoy-federatsii-ot-3-avgusta-2012-g-66n-ob-utverzhdenii-poryadka-i-srokov-sovershenstvovaniya-meditsinskimi-rabotnikami-i-farmatsevticheskimi-rabotnikami-professionalnyh-znaniy-i-navykov-putem-obucheniya-po-dopolnitelnym-professionalnym-obrazovatelnym-programmam-v-obrazovatelnyh-i-nauchnyh-organizatsiyah",
        "https://docs.cntd.ru/document/420310213",
        "https://normativ.kontur.ru/document?moduleId=1&documentId=269836",
        "https://mosgorzdrav.ru/ru-RU/moscowDoctorDetail.html",
        "https://normativ.kontur.ru/document?moduleId=1&documentId=457082",
        "http://publication.pravo.gov.ru/Document/View/0001202101120007?index=2"
    ]
    
    main(urls)