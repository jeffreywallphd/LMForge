from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render

import requests
import json
from bs4 import BeautifulSoup
from openpyxl import load_workbook
from io import BytesIO
import pdfplumber
import markdown
import logging

from ..models.scraped_data import ScrapedData

logger = logging.getLogger(__name__)


class ScrapeDataView(APIView):
    """Scrape data from a URL and save to ScrapedData."""

    def get(self, request):
        url = request.GET.get("url")
        title = request.GET.get("title", "")
        if not url:
            return Response({"error": "Missing 'url' parameter"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
        except Exception as e:
            logger.exception("Failed to fetch url %s", url)
            return Response({"error": f"Failed to fetch URL: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        ctype = r.headers.get("content-type", "").lower()

        # JSON
        if "application/json" in ctype or (r.text and r.text.strip().startswith("{")):
            content = json.dumps(r.json(), indent=2)
            file_type = "json"

        # XML-ish
        elif any(x in ctype for x in ("application/xml", "text/xml", "application/rss+xml")):
            content = r.text
            file_type = "xml"

        # Plain text
        elif "text/plain" in ctype:
            content = r.text
            file_type = "text"

        # CSV
        elif "text/csv" in ctype or url.lower().endswith(".csv"):
            content = r.text
            file_type = "csv"

        # XLSX / Excel
        elif any(x in ctype for x in ("excel", "spreadsheetml", "vnd.openxmlformats")) or url.lower().endswith(".xlsx"):
            try:
                bio = BytesIO(r.content)
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
            # Basic HTML extraction (kept intentionally simple here; a dedicated extractor will be used later)
            soup = BeautifulSoup(r.content, "html.parser")
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
            sd = ScrapedData.objects.create(
                url=url,
                file_type=file_type,
                content=content,
                title=title or (url[:95] if url else "scraped")
            )
        except Exception as e:
            logger.exception("Failed to save ScrapedData: %s", e)
            return Response({"error": "Failed to save scraped data"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"success": "Scraped and saved", "id": sd.id, "url": sd.url, "file_type": sd.file_type, "content": sd.content})


class UploadPDFView(APIView):
    """Upload a PDF and convert it to text/html/json as requested."""

    def post(self, request):
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

            sd = ScrapedData.objects.create(
                url="uploaded_pdf",
                file_type=file_type,
                content=content,
                pdf_file=pdf_file,
                title=title[:100]
            )
        except Exception as e:
            logger.exception("PDF conversion failed: %s", e)
            return Response({"error": "Failed to convert PDF"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"success": "PDF converted and saved", "id": sd.id, "file_type": sd.file_type, "content": sd.content})


def scrape_view(request):
    latest = ScrapedData.objects.order_by("-created_at").first()
    return render(request, "scrape.html", {"scraped": latest})


class SaveManualTextView(APIView):
    def post(self, request):
        text = request.data.get("text") or request.POST.get("text")
        title = request.data.get("title") or request.POST.get("title") or "manual"
        if not text:
            return Response({"error": "No text provided"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            sd = ScrapedData.objects.create(
                url="manual",
                file_type="text",
                content=text,
                title=title[:100]
            )
        except Exception as e:
            logger.exception("Failed to save manual text: %s", e)
            return Response({"error": "Failed to save text"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response({"success": "Text saved", "id": sd.id, "file_type": sd.file_type})