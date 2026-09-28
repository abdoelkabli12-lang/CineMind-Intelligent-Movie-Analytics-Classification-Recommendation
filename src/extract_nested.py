import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

BASE_URL = 'https://api.themoviedb.org/3'
API_KEY = os.getenv('TMDB_API_KEY')

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

def extract_films(min_id=1, max_id=100, output_file='data/raw/all_movies.json'):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    all_movies = []
    errors = []
    
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
                errors.append({'id': movie_id, 'error': 'not_found'})
                continue
            
            response.raise_for_status()
            full_data = response.json()
            
            filtered = {}
            for tmdb_field, your_field in FIELDS_TO_KEEP.items():
                filtered[your_field] = full_data.get(tmdb_field)
            
            all_movies.append(filtered)
            print(f"  + ID {movie_id}: {filtered.get('title', 'Unknown')}")
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"  ID {movie_id}: rate limited, waiting...")
                time.sleep(5)
                continue
            errors.append({'id': movie_id, 'error': f'http_{response.status_code}'})
            print(f"  ID {movie_id}: HTTP error {response.status_code}")
        except Exception as e:
            errors.append({'id': movie_id, 'error': str(e)})
            print(f"  ID {movie_id}: error - {e}")
        
        time.sleep(0.25)
    
    output = {
        'metadata': {
            'total_fetched': len(all_movies),
            'total_errors': len(errors),
            'range': {'min_id': min_id, 'max_id': max_id},
            'fields': list(FIELDS_TO_KEEP.values())
        },
        'movies': all_movies,
        'errors': errors
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nDone! {len(all_movies)} movies saved to {output_file}")
    print(f"Errors: {len(errors)}")

extract_films(min_id=1, max_id=100)
