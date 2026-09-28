import requests
import json
from dotenv import load_dotenv
import os
load_dotenv()
import time

BASE_URL = 'https://api.themoviedb.org/3'
API_KEY = os.getenv('TMDB_API_KEY')

# def get_movie_details(movie_id = 2):
#   url = f'{BASE_URL}/movie/{movie_id}'
#   params = {
#     'api_key': API_KEY,
#     'language': 'en-US',
#     'append_to_response': 'keywords,genres',
#     'movie_id' : movie_id,
#   }
#   response = requests.get(url, params= params, timeout=10)
#   print(f"Status: {response.status_code}")
#   print(f"Response: {response.text[:300]}")  # see what TMDB actually sent back
#   response.raise_for_status()
#   return response.json()


# def search_popular_movies(page=1, per_page=20):
#   url = f'{BASE_URL}/movie/popular'
#   params = {
#     'api_key': API_KEY,
#       'language': 'en-US',
#       'append_to_response': 'keywords,genres',
#       'page': page
#   }
#   response = requests.get(url, params= params, timeout=10)
#   print(f"Status: {response.status_code}")
#   print(f"Response: {response.text[:300]}")  # see what TMDB actually sent back
#   response.raise_for_status()
#   data = response.json()
  
#   results = data["results"][:per_page]
  
#   return {
#         "page": data["page"],
#         "per_page": per_page,
#         "total_pages": data["total_pages"],
#         "total_results": data["total_results"],
#         "results": [
#             {
#                 "id": m["id"],
#                 "title": m["title"],
#                 "overview": m["overview"],
#                 "release_date": m["release_date"],
#                 "vote_average": m["vote_average"],
#                 "vote_count": m["vote_count"],
#                 "poster_path": f"https://image.tmdb.org/t/p/w500{m['poster_path']}",
#             }
#             for m in results
#         ],
#     }


# data = search_popular_movies(page=1, per_page=20)
# for movie in data["results"]:
#     print(f"{movie['title']} ({movie['release_date'][:4]}) — ★ {movie['vote_average']}")
    
    
    
FIELDS_TO_KEEP = {
    'id': 'movie_id',
    'title': 'title',
    'overview': 'overview',
    'release_date': 'release_date',
    'runtime': 'runtime',
    'original_language': 'original_language',
    'genres': 'genres',
    'keywords': 'keywords',
    'budget': 'budget',
    'revenue': 'revenue',
    'popularity': 'popularity',
    'vote_average': 'vote_average',
    'vote_count': 'vote_count',
}
    
    
    
def extract_films(min_id=1, max_id=4000, output_file='data/raw/all_movies.json'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    all_movies = []
    
    for movie_id in range(min_id, max_id + 1):
        url = f'{BASE_URL}/movie/{movie_id}'
        params = {
            'api_key': API_KEY,
            'language': 'en-US',
            'append_to_response': 'keywords,genres'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 404:
                continue
            
            response.raise_for_status()
            full_data = response.json()
            
            # Filter to only the fields you want
            filtered = {}
            for tmdb_field, your_field in FIELDS_TO_KEEP.items():
                filtered[your_field] = full_data.get(tmdb_field)
            
            all_movies.append(filtered)
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                time.sleep(5)
                continue
        except Exception:
            pass
        
        time.sleep(0.25)
        
        # Progress print every 50 movies
        if movie_id % 50 == 0:
            print(f"  Progress: {movie_id}/{max_id} ({len(all_movies)} movies fetched)")
    
    output = {
        'metadata': {
            'total_fetched': len(all_movies),
            'range': {'min_id': min_id, 'max_id': max_id},
            'fields': list(FIELDS_TO_KEEP.values())
        },
        'movies': all_movies
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! {len(all_movies)} movies saved to {output_file}")
# Run it
extract_films(min_id=1, max_id=100)  # Start small to test, then increase