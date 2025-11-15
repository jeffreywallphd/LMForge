from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.request import Request
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse

import requests
import json
from bs4 import BeautifulSoup
from openpyxl import load_workbook
from io import BytesIO
import pdfplumber
import markdown
import logging
from typing import Dict, Any

from ..models.scraped_data import ScrapedData
from ..utils.content_extractor import extract_article_content

logger = logging.getLogger(__name__)

# Constants
MAX_TITLE_LENGTH = 100  # Maximum length for ScrapedData title field
MAX_URL_TITLE_LENGTH = 95  # Maximum length when using URL as title (leave room for "scraped")


class ScrapeDataView(APIView):
    """Scrape data from a URL and save to ScrapedData."""

    def get(self, request: Request) -> Response:
        url: str | None = request.GET.get("url")
        title: str = request.GET.get("title", "")
        if not url:
            return Response({"error": "Missing 'url' parameter"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
        except Exception as e:
            logger.exception("Failed to fetch url %s", url)
            return Response({"error": f"Failed to fetch URL: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        content_type: str = response.headers.get("content-type", "").lower()

        # JSON
        if "application/json" in content_type or (response.text and response.text.strip().startswith("{")):
            content = json.dumps(response.json(), indent=2)
            file_type = "json"

        # XML-ish
        elif any(x in content_type for x in ("application/xml", "text/xml", "application/rss+xml")):
            content = response.text
            file_type = "xml"

        # Plain text
        elif "text/plain" in content_type:
            content = response.text
            file_type = "text"

        # CSV
        elif "text/csv" in content_type or url.lower().endswith(".csv"):
            content = response.text
            file_type = "csv"

        # XLSX / Excel
        elif any(x in content_type for x in ("excel", "spreadsheetml", "vnd.openxmlformats")) or url.lower().endswith(".xlsx"):
            try:
                bio = BytesIO(response.content)
                wb = load_workbook(filename=bio, read_only=True)
                ws = wb[wb.sheetnames[0]]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    rows.append(",".join([str(c) if c is not None else "" for c in row]))
                content = "\n".join(rows)
                file_type = "xlsx"
            except Exception as e:
                logger.exception("Failed to parse xlsx: %s", e)
                return Response({"error": "Failed to parse xlsx file"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # HTML
        else:
            # Use the centralized extractor; fall back to basic BeautifulSoup parsing on error
            try:
                result: Dict[str, Any] = extract_article_content(response.content, url)
                content = result.get("body", "") or ""

                # If extractor returned an empty body, fall back to simple parsing
                if not content.strip():
                    logger.info("Extractor returned empty body for %s; falling back to simple HTML parsing", url)
                    soup = BeautifulSoup(response.content, "html.parser")
                    article = soup.find("article") or soup.find(class_="content") or soup.find("main")
                    if article:
                        text = article.get_text("\n\n", strip=True)
                    else:
                        body = soup.body
                        text = body.get_text("\n\n", strip=True) if body else soup.get_text("\n\n", strip=True)
                    content = text

                # If user did not provide a title, prefer the extracted one
                if not title or not title.strip():
                    extracted_title = result.get("title") or ""
                    if extracted_title:
                        title = extracted_title
                file_type = "html"
            except Exception as e:
                logger.exception("Extractor failed, falling back to simple HTML parsing: %s", e)
                soup = BeautifulSoup(response.content, "html.parser")
                article = soup.find("article") or soup.find(class_="content") or soup.find("main")
                if article:
                    text = article.get_text("\n\n", strip=True)
                else:
                    body = soup.body
                    text = body.get_text("\n\n", strip=True) if body else soup.get_text("\n\n", strip=True)
                content = text
                file_type = "html"

        # Save
        try:
            scraped_record: ScrapedData = ScrapedData.objects.create(
                url=url,
                file_type=file_type,
                content=content,
                title=title or (url[:MAX_URL_TITLE_LENGTH] if url else "scraped")
            )
        except Exception as e:
            logger.exception("Failed to save ScrapedData: %s", e)
            return Response({"error": "Failed to save scraped data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "success": "Scraped and saved", 
            "id": scraped_record.id, 
            "url": scraped_record.url, 
            "file_type": scraped_record.file_type, 
            "content": scraped_record.content
        })


class UploadPDFView(APIView):
    """Upload a PDF and convert it to text/html/json as requested."""

    def post(self, request: Request) -> Response:
        pdf_file = request.FILES.get("pdf_file")
        output_format = request.POST.get("output_format") or request.data.get("output_format") or "text"
        title = request.POST.get("title") or request.data.get("title") or (getattr(pdf_file, 'name', '') if pdf_file else "uploaded_pdf")

        if not pdf_file:
            return Response({"error": "No PDF file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            text_parts = []
            with pdfplumber.open(pdf_file) as pdf:
                for p in pdf.pages:
                    text_parts.append(p.extract_text() or "")
            text = "\n\n".join([t for t in text_parts if t])

            if output_format == "html":
                content = markdown.markdown(text)
                file_type = "html"
            elif output_format == "json":
                content = json.dumps({"text": text}, indent=2)
                file_type = "json"
            else:
                content = text
                file_type = "text"

            scraped_record: ScrapedData = ScrapedData.objects.create(
                url="uploaded_pdf",
                file_type=file_type,
                content=content,
                pdf_file=pdf_file,
                title=title[:MAX_TITLE_LENGTH]
            )
        except Exception as e:
            logger.exception("PDF conversion failed: %s", e)
            return Response({"error": "Failed to convert PDF"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({
            "success": "PDF converted and saved", 
            "id": scraped_record.id, 
            "file_type": scraped_record.file_type, 
            "content": scraped_record.content
        })


def scrape_view(request: HttpRequest) -> HttpResponse:
    """Render the scrape view with the latest scraped data."""
    latest: ScrapedData | None = ScrapedData.objects.order_by("-created_at").first()
    return render(request, "scrape.html", {"scraped": latest})


class SaveManualTextView(APIView):
    """Save manually entered text to ScrapedData."""
    
    def post(self, request: Request) -> Response:
        text: str | None = request.data.get("text") or request.POST.get("text")
        title: str = request.data.get("title") or request.POST.get("title") or "manual"
        if not text:
            return Response({"error": "No text provided"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            scraped_record: ScrapedData = ScrapedData.objects.create(
                url="manual",
                file_type="text",
                content=text,
                title=title[:MAX_TITLE_LENGTH]
            )
        except Exception as e:
            logger.exception("Failed to save manual text: %s", e)
            return Response({"error": "Failed to save text"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({"success": "Text saved", "id": scraped_record.id, "file_type": scraped_record.file_type})