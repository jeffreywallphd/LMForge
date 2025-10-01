from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import requests
import json
from bs4 import BeautifulSoup
from django.shortcuts import render
from openpyxl import load_workbook
from io import BytesIO
from ..models.scraped_data import ScrapedData  # Import the model to save data
import pdfplumber
import markdown
from transformers import pipeline
# Make sure to import urlparse
from urllib.parse import urlparse
import time

from django.conf import settings
from django.conf.urls.static import static

class ScrapeDataView(APIView):
    def get(self, request):
        url = request.GET.get('url')
        title = request.GET.get('title')
        # --- MODIFICATION 1: Get the new source_type parameter ---
        source_type = request.GET.get('source_type') # e.g., 'reddit' or 'generic'

        if not url:
            return Response({'error': 'Please provide a URL.'}, status=status.HTTP_400_BAD_REQUEST)

        # Initialize variables for content storage
        scraped_content = None
        binary_content = None
        file_type = None

        # --- MODIFICATION 2: Main logic now checks for the source_type ---
        if source_type == 'reddit':
            # --- START: Reddit API Integration ---
            parsed_url = urlparse(url)
            # Convert the regular Reddit URL to its JSON API equivalent
            api_url = url.rstrip('/') + ".json"

            try:
                # Reddit's API requires a custom User-Agent header
                headers = {'User-Agent': 'My-Django-Scraper-App/1.2'}
                response = requests.get(api_url, headers=headers)
                response.raise_for_status()
                data = response.json()

                # LOGIC 1: Handle a specific post URL (contains /comments/)
                if '/comments/' in parsed_url.path:
                    file_type = 'reddit_post'
                    
                    post_data = data[0]['data']['children'][0]['data']
                    post_title = post_data.get('title', 'No Title')
                    post_author = post_data.get('author', 'Unknown Author')
                    post_text = post_data.get('selftext', 'No content text.')
                    
                    content_lines = [
                        f"Title: {post_title}",
                        f"Author: u/{post_author}",
                        "--- POST CONTENT ---",
                        post_text,
                        "\n--- COMMENTS ---"
                    ]

                    comments_data = data[1]['data']['children']
                    for comment in comments_data:
                        if 'data' in comment and 'body' in comment['data']:
                            comment_author = comment['data'].get('author', 'Unknown')
                            comment_body = comment['data'].get('body', '')
                            content_lines.append(f"\n> u/{comment_author}:\n{comment_body}\n")
                    
                    scraped_content = "\n".join(content_lines)

                # LOGIC 2: Handle a subreddit URL (e.g., /r/Python/)
                elif parsed_url.path.startswith('/r/'):
                    file_type = 'reddit_subreddit_full'
                    
                    subreddit_name = parsed_url.path.split('/')[2]
                    posts = data['data']['children']
                    
                    content_lines = [f"Scraped Posts and Comments from r/{subreddit_name}:\n" + "="*40]

                    for i, post_item in enumerate(posts):
                        post_data = post_item['data']
                        post_title = post_data.get('title', 'No Title')
                        post_author = post_data.get('author', 'Unknown')
                        permalink = post_data.get('permalink')

                        if not permalink:
                            continue

                        content_lines.append(f"\n\n--- POST {i+1}: {post_title} (by u/{post_author}) ---\n")
                        post_api_url = f"https://www.reddit.com{permalink.rstrip('/')}.json"
                        
                        try:
                            post_response = requests.get(post_api_url, headers=headers)
                            post_response.raise_for_status()
                            post_and_comment_data = post_response.json()

                            post_content_data = post_and_comment_data[0]['data']['children'][0]['data']
                            post_text = post_content_data.get('selftext', '')
                            if post_text:
                                content_lines.append(f"--- POST CONTENT ---\n{post_text}\n")

                            content_lines.append("--- COMMENTS ---")
                            comments_data = post_and_comment_data[1]['data']['children']
                            if not comments_data:
                                content_lines.append("No comments found for this post.")
                            else:
                                for comment in comments_data:
                                    if 'data' in comment and 'body' in comment['data']:
                                        comment_author = comment['data'].get('author', 'Unknown')
                                        comment_body = comment['data'].get('body', '')
                                        content_lines.append(f"\n> u/{comment_author}:\n{comment_body}\n")
                            
                            time.sleep(0.5)

                        except requests.RequestException as post_e:
                            content_lines.append(f"\n[Could not fetch content for this post. Error: {str(post_e)}]")
                        except (KeyError, IndexError):
                            content_lines.append("\n[Could not parse content for this post.]")
                    
                    scraped_content = "\n".join(content_lines)

                # LOGIC 3: Invalid Reddit URL
                else:
                    return Response({
                        'error': 'This appears to be a Reddit URL, but not a valid post or subreddit page.'
                    }, status=status.HTTP_400_BAD_REQUEST)

            except requests.RequestException as e:
                return Response({'error': f'Failed to retrieve data from Reddit API. Error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
                 return Response({'error': f'Failed to parse Reddit API response. Ensure the URL is valid. Error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            # --- END: Reddit API Integration ---

        else:
            # --- This is the original code for handling all other non-Reddit URLs ---
            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.RequestException as e:
                return Response({'error': f'Failed to retrieve data from the URL. Error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

            content_type = response.headers.get('Content-Type', '').lower()

            if 'application/json' in content_type:
                file_type = 'json'
                scraped_content = json.dumps(response.json(), indent=4)
            elif 'application/xml' in content_type or 'text/xml' in content_type:
                file_type = 'xml'
                scraped_content = response.content.decode('utf-8')
            elif 'text/plain' in content_type:
                file_type = 'text'
                scraped_content = response.content.decode('utf-8')
            elif 'text/html' in content_type:
                file_type = 'html'
                soup = BeautifulSoup(response.content, 'html.parser')
                for script in soup(["script", "style", "meta", "noscript"]):
                    script.extract()
                main_content = soup.find("article")
                if not main_content:
                    main_content = soup.find("div", {"class": "content"})
                if main_content:
                    scraped_content = main_content.get_text(separator="\n", strip=True)
                else:
                    scraped_content = soup.get_text(separator="\n", strip=True)
                scraped_content = "\n".join([line.strip() for line in scraped_content.split("\n") if line.strip()])
            elif 'text/csv' in content_type or 'application/csv' in content_type:
                file_type = 'csv'
                scraped_content = response.content.decode('utf-8')
            elif 'application/vnd.ms-excel' in content_type or 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
                file_type = 'xlsx'
                binary_content = response.content
            else:
                return Response({'error': f'Unsupported content type: {content_type}'}, status=status.HTTP_400_BAD_REQUEST)

        # Save the scraped data to the database
        ScrapedData.objects.create(
            url=url,
            file_type=file_type,
            content=scraped_content,
            binary_content=binary_content,
            title=title
        )

        latest_scraped_data = ScrapedData.objects.latest('created_at')

        return Response({
            'success': f'Successfully saved data from {url} to the database.',
            'url': latest_scraped_data.url,
            'file_type': latest_scraped_data.file_type,
            'content': latest_scraped_data.content
        }, status=status.HTTP_200_OK)

class UploadPDFView(APIView):
    def post(self, request):
        pdf_file = request.FILES.get('pdf_file')
        output_format = request.POST.get('output_format')
        title = request.POST.get('title')

        if not pdf_file or not output_format:
            return Response({'error': 'PDF file and output format are required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            extracted_text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() + "\n\n"

            file_type = ''
            converted_content = ''

            if output_format == 'html':
                converted_content = markdown.markdown(extracted_text)
                file_type = 'html'
            elif output_format == 'json':
                json_content = {"content": extracted_text.strip().split('\n')}
                converted_content = json.dumps(json_content, indent=4) # Ensure JSON is stored as a string
                file_type = 'json'
            elif output_format == 'text':
                converted_content = extracted_text.strip()
                file_type = 'text'
            else:
                return Response({'error': 'Unsupported output format.'}, status=status.HTTP_400_BAD_REQUEST)

            scraped_data = ScrapedData.objects.create(
                file_type=file_type,
                content=converted_content,
                title=title
            )

            latest_scraped_data = ScrapedData.objects.latest('created_at')

            return Response({
                'success': f'PDF successfully converted to {output_format.upper()}.',
                'file_type': file_type,
                'content': converted_content,
                'latest_scraped_data': {
                    'url': latest_scraped_data.url,
                    'file_type': latest_scraped_data.file_type,
                    'content': latest_scraped_data.content,
                }
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': f'Error processing PDF: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def scrape_view(request):
    try:
        latest_scraped_data = ScrapedData.objects.latest('created_at')
    except ScrapedData.DoesNotExist:
        latest_scraped_data = None
    return render(request, 'scrape.html', {'latest_scraped_data': latest_scraped_data})

class SaveManualTextView(APIView):
    def post(self, request):
        text = request.data.get('text')
        title = request.data.get('title')
        if not text:
            return Response({'error': 'Please provide text.'}, status=status.HTTP_400_BAD_REQUEST)

        ScrapedData.objects.create(
            file_type='text',
            content=text,
            title=title
        )

        latest_scraped_data = ScrapedData.objects.latest('created_at')

        return Response({
            'success': 'Successfully saved manually entered text.',
            'file_type': latest_scraped_data.file_type,
            'content': latest_scraped_data.content
        }, status=status.HTTP_200_OK)