import logging
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import time
import os
import hashlib
import sqlite3
import datetime
from contextlib import closing
import random

# Cache directory for search results
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "search")
os.makedirs(CACHE_DIR, exist_ok=True)

# SQLite database for caching
CACHE_DB = os.path.join(CACHE_DIR, "search_cache.db")

# Initialize the cache database
def _init_cache_db():
    with closing(sqlite3.connect(CACHE_DB)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_cache (
                query_hash TEXT PRIMARY KEY,
                query TEXT,
                results TEXT,
                timestamp TIMESTAMP
            )
            ''')
            conn.commit()

# Initialize the cache database at module load time
_init_cache_db()

def search_literature(query: str, max_results: int = 5, use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Searches scientific literature using multiple real sources for the given query.
    IMPORTANT: This function must use real APIs and should never generate synthetic papers.
    
    The following sources are tried in order:
    1. PubMed via NCBI's E-utilities API
    2. arXiv via their official API
    3. Semantic Scholar as a backup source
    
    Args:
        query: The search query (research goal or topic)
        max_results: Maximum number of results to return
        use_cache: Whether to use and update the cache
        
    Returns:
        List of paper dictionaries with title, authors, abstract, etc.
    """
    logging.info(f"Searching literature for: {query}")
    
    # Generate a hash for the query to use as cache key
    query_hash = hashlib.md5(query.encode()).hexdigest()
    
    # Check if results are in cache
    if use_cache:
        cached_results = _get_from_cache(query_hash, query)
        if cached_results:
            logging.info(f"Using cached results for query: {query}")
            return cached_results[:max_results]
    
    # Try PubMed first
    try:
        results = _search_pubmed(query, max_results)
        if results and len(results) >= max_results // 2:
            # Update cache
            if use_cache:
                _save_to_cache(query_hash, query, results)
            
            return results[:max_results]
    except Exception as e:
        logging.error(f"Error searching PubMed: {e}")
        results = []
    
    # Try arXiv as a backup source
    if not results or len(results) < max_results // 2:
        try:
            arxiv_results = _search_arxiv(query, max_results)
            if arxiv_results:
                results.extend(arxiv_results)
                results = _deduplicate_results(results)
        except Exception as e:
            logging.error(f"Error searching arXiv: {e}")
    
    # If we still don't have enough results, try a semantic scholar API
    if not results or len(results) < max_results // 2:
        try:
            semantic_results = _search_semantic_scholar(query, max_results)
            if semantic_results:
                results.extend(semantic_results)
                results = _deduplicate_results(results)
        except Exception as e:
            logging.error(f"Error searching Semantic Scholar: {e}")
    
    # If all sources fail, try the cache regardless of the use_cache parameter
    if not results:
        cached_results = _get_from_cache(query_hash, query, ignore_age=True)
        if cached_results:
            logging.warning(f"All sources failed, using cached results for query: {query}")
            return cached_results[:max_results]
    
    # Return empty list if no results found (instead of raising an error)
    if not results:
        logging.warning(f"No literature found for the research goal: {query}. Continuing with empty results.")
        return []
    
    # Update cache with results
    if use_cache and results:
        _save_to_cache(query_hash, query, results)
    
    return results[:max_results]

def _get_from_cache(query_hash: str, query: str, max_age_hours: int = 24, ignore_age: bool = False) -> List[Dict[str, Any]]:
    """Get results from cache if available and not too old."""
    try:
        with closing(sqlite3.connect(CACHE_DB)) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute(
                    "SELECT results, timestamp FROM search_cache WHERE query_hash = ?", 
                    (query_hash,)
                )
                result = cursor.fetchone()
                
                if result:
                    results_json, timestamp_str = result
                    timestamp = datetime.datetime.fromisoformat(timestamp_str)
                    age = datetime.datetime.now() - timestamp
                    
                    # Return if results are fresh enough or we're ignoring age
                    if ignore_age or age.total_seconds() < max_age_hours * 3600:
                        return json.loads(results_json)
                    else:
                        logging.info(f"Cached results for '{query}' are too old ({age.total_seconds()/3600:.1f} hours)")
    except Exception as e:
        logging.error(f"Error accessing cache: {e}")
    
    return []

def _save_to_cache(query_hash: str, query: str, results: List[Dict[str, Any]]):
    """Save results to cache."""
    try:
        with closing(sqlite3.connect(CACHE_DB)) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute(
                    "INSERT OR REPLACE INTO search_cache VALUES (?, ?, ?, ?)",
                    (
                        query_hash,
                        query,
                        json.dumps(results),
                        datetime.datetime.now().isoformat()
                    )
                )
                conn.commit()
    except Exception as e:
        logging.error(f"Error saving to cache: {e}")

def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate papers from search results based on title similarity."""
    if not results:
        return []
    
    deduplicated = []
    seen_titles = set()
    
    for paper in results:
        title = paper.get("title", "").lower()
        # Generate a simplified title for comparison
        simple_title = ''.join(c for c in title if c.isalnum())
        
        # Check if we've seen a very similar title
        is_duplicate = False
        for seen_title in seen_titles:
            # Check for high similarity
            if len(simple_title) > 0 and len(seen_title) > 0:
                shorter = min(len(simple_title), len(seen_title))
                similarity = sum(a == b for a, b in zip(simple_title[:shorter], seen_title[:shorter])) / shorter
                if similarity > 0.9:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_titles.add(simple_title)
            deduplicated.append(paper)
    
    return deduplicated

def _search_pubmed(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search PubMed API for scientific papers."""
    # Step 1: Search PubMed for IDs
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance"
    }
    
    search_url_with_params = f"{search_url}?{urllib.parse.urlencode(search_params)}"
    
    with urllib.request.urlopen(search_url_with_params) as response:
        search_results = json.loads(response.read().decode())
        
    if "esearchresult" not in search_results or "idlist" not in search_results["esearchresult"]:
        logging.warning(f"No search results found in PubMed for query: {query}")
        return []
        
    id_list = search_results["esearchresult"]["idlist"]
    
    if not id_list:
        logging.warning(f"Empty ID list returned from PubMed for query: {query}")
        return []
        
    # Step 2: Fetch details for the IDs
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    
    fetch_url_with_params = f"{fetch_url}?{urllib.parse.urlencode(fetch_params)}"
    
    with urllib.request.urlopen(fetch_url_with_params) as response:
        xml_data = response.read().decode()
        
    # Step 3: Parse XML and extract paper details
    papers = _parse_pubmed_xml(xml_data)
    
    # Add a slight delay to respect API rate limits
    time.sleep(0.34)  # Respect NCBI's limit of 3 requests per second
    
    logging.info(f"Found {len(papers)} papers in PubMed related to '{query}'")
    
    # Log the actual paper titles that were fetched
    for i, paper in enumerate(papers, 1):
        title = paper.get('title', 'Untitled')
        authors = paper.get('authors', [])
        author_text = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '') if authors else 'No authors'
        logging.info(f"  PubMed Paper {i}: {title}")
        logging.info(f"    Authors: {author_text}")
        logging.info(f"    PMID: {paper.get('pmid', 'N/A')}")
    
    return papers

def _search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search arXiv API for scientific papers."""
    arxiv_url = "http://export.arxiv.org/api/query"
    
    # Clean the query for arXiv
    clean_query = query.replace(':', ' ').replace('-', ' ').replace('/', ' ')
    
    # Build the arXiv search URL
    params = {
        "search_query": f"all:{clean_query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    url = f"{arxiv_url}?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url) as response:
            xml_data = response.read().decode()
            
        # Parse arXiv XML
        root = ET.fromstring(xml_data)
        papers = []
        
        # arXiv namespace
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        for entry in root.findall(".//atom:entry", ns):
            try:
                # Extract paper details
                title_element = entry.find("atom:title", ns)
                title = title_element.text if title_element is not None else "Untitled"
                
                # Extract authors
                authors = []
                for author in entry.findall(".//atom:author/atom:name", ns):
                    if author.text:
                        authors.append(author.text)
                
                # Extract abstract
                summary_element = entry.find("atom:summary", ns)
                abstract = summary_element.text if summary_element is not None else "No abstract available."
                
                # Extract URL
                url_element = entry.find(".//atom:link[@title='pdf']", ns)
                url = url_element.get("href") if url_element is not None else ""
                
                # Extract publication date
                published_element = entry.find("atom:published", ns)
                year = None
                if published_element is not None and published_element.text:
                    try:
                        year = int(published_element.text[:4])  # Extract year from ISO date format
                    except:
                        pass
                
                paper = {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "journal": "arXiv",
                    "year": year,
                    "url": url
                }
                papers.append(paper)
            except Exception as e:
                logging.warning(f"Error parsing arXiv entry: {e}")
                continue
        
        logging.info(f"Found {len(papers)} papers in arXiv related to '{query}'")
        
        # Log the actual paper titles that were fetched
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Untitled')
            authors = paper.get('authors', [])
            author_text = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '') if authors else 'No authors'
            logging.info(f"  arXiv Paper {i}: {title}")
            logging.info(f"    Authors: {author_text}")
            logging.info(f"    Year: {paper.get('year', 'N/A')}")
        
        return papers
    except Exception as e:
        logging.error(f"Error searching arXiv: {e}")
        return []

def _search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Semantic Scholar API for scientific papers."""
    # Note: This is a basic implementation. A full version would use Semantic Scholar's API key.
    api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,venue,year,url"
    }
    
    request_url = f"{api_url}?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(
            request_url,
            headers={
                'User-Agent': 'Mozilla/5.0 Python/3.8',
                'Accept': 'application/json'
            }
        )
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        papers = []
        for item in data.get("data", []):
            author_list = []
            for author in item.get("authors", []):
                name = author.get("name")
                if name:
                    author_list.append(name)
            
            paper = {
                "title": item.get("title", "Untitled"),
                "authors": author_list,
                "abstract": item.get("abstract", "No abstract available."),
                "journal": item.get("venue", "Unknown"),
                "year": item.get("year"),
                "url": item.get("url", "")
            }
            papers.append(paper)
            
        logging.info(f"Found {len(papers)} papers in Semantic Scholar related to '{query}'")
        
        # Log the actual paper titles that were fetched
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Untitled')
            authors = paper.get('authors', [])
            author_text = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '') if authors else 'No authors'
            logging.info(f"  Semantic Scholar Paper {i}: {title}")
            logging.info(f"    Authors: {author_text}")
            logging.info(f"    Journal: {paper.get('journal', 'N/A')}")
            logging.info(f"    Year: {paper.get('year', 'N/A')}")
        
        return papers
    except Exception as e:
        logging.error(f"Error searching Semantic Scholar: {e}")
        return []

def _parse_pubmed_xml(xml_data: str) -> List[Dict[str, Any]]:
    """
    Parse PubMed XML response and extract paper details.
    """
    papers = []
    
    try:
        root = ET.fromstring(xml_data)
        articles = root.findall(".//PubmedArticle")
        
        for article in articles:
            try:
                # Extract article metadata
                article_data = {}
                
                # Title
                title_element = article.find(".//ArticleTitle")
                if title_element is not None and title_element.text:
                    article_data["title"] = title_element.text
                else:
                    continue  # Skip articles without titles
                
                # Authors
                authors = []
                author_elements = article.findall(".//Author")
                for author_element in author_elements:
                    last_name = author_element.find("LastName")
                    first_name = author_element.find("ForeName")
                    if last_name is not None and last_name.text:
                        author_name = last_name.text
                        if first_name is not None and first_name.text:
                            author_name = f"{first_name.text} {author_name}"
                        authors.append(author_name)
                
                article_data["authors"] = authors
                
                # Abstract
                abstract_elements = article.findall(".//AbstractText")
                abstract_parts = []
                for abstract_element in abstract_elements:
                    if abstract_element.text:
                        abstract_parts.append(abstract_element.text)
                
                if abstract_parts:
                    article_data["abstract"] = " ".join(abstract_parts)
                else:
                    article_data["abstract"] = "No abstract available."
                
                # Journal info
                journal_element = article.find(".//Journal/Title")
                if journal_element is not None and journal_element.text:
                    article_data["journal"] = journal_element.text
                
                # Publication year
                year_element = article.find(".//PubDate/Year")
                if year_element is not None and year_element.text:
                    try:
                        article_data["year"] = int(year_element.text)
                    except ValueError:
                        article_data["year"] = None
                
                # PMID for URL
                pmid_element = article.find(".//PMID")
                if pmid_element is not None and pmid_element.text:
                    article_data["pmid"] = pmid_element.text
                    article_data["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_element.text}/"
                
                # Add to papers list
                papers.append(article_data)
                
            except Exception as e:
                logging.warning(f"Error parsing article: {e}")
                continue
                
        return papers
        
    except Exception as e:
        logging.error(f"Error parsing PubMed XML: {e}")
        return []

def perform_web_search(query: str) -> str:
    """
    Performs a web search for scientific literature using real academic APIs.
    
    Args:
        query: The search query to find relevant academic articles
        
    Returns:
        A string containing formatted search results
    """
    logging.info(f"Performing web search for query: {query}")
    
    try:
        # Use the real search_literature function that connects to PubMed, arXiv, etc.
        papers = search_literature(query, max_results=5)
        
        # Handle case where no papers were found
        if not papers:
            return "No relevant papers found for this query. Continuing without literature."
        
        # Format the results into the expected string format
        results = []
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', f"Untitled Paper {i}")
            authors = paper.get('authors', [])
            authors_text = ', '.join(authors) if authors else "No authors listed"
            year = paper.get('year', "N/A")
            journal = paper.get('journal', "Unknown Journal")
            url = paper.get('url', "")
            
            # Create a snippet from the abstract if available
            abstract = paper.get('abstract', "")
            snippet = abstract[:150] + "..." if abstract and len(abstract) > 150 else abstract
            
            # Format similar to the previous version for compatibility
            result = f"{i}. {title}\n   {authors_text} ({year}). {journal}."
            if url:
                result += f" URL: {url}"
            result += f"\n   {snippet}\n"
            
            results.append(result)
        
        return "\n".join(results) if results else "No relevant papers found for this query."
        
    except Exception as e:
        logging.error(f"Error performing web search: {e}")
        return f"Error performing search, continuing without literature: {str(e)}" 