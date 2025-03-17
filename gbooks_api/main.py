'''
Note: the API calls are all done in French language only
'''

# imports

import time
import csv
import requests
import os
import pandas as pd

# constants

KEY_BOOK_INFO_LIST = [
        "Title", "Authors", "Publisher", "Published Date", "Categories",
        "Description", "Page Count", "Language", "ISBN-10", "ISBN-13", "Preview Link", "Info Link",
        "Average Rating", "Ratings Count", "Image Link", "Saleability", "Price", "Currency"
    ]

# internal functions

def database_row_inputer(book: dict) -> list:
    '''
    From a book from the API call
    '''

    # looking up for each information

    volume_info = book.get("volumeInfo", {})
    sale_info = book.get("saleInfo", {})
    title = volume_info.get("title", "Titre inconnu")
    authors = ", ".join(volume_info.get("authors", ["Auteur inconnu"]))
    publisher = volume_info.get("publisher", "Éditeur inconnu")
    published_date = volume_info.get("publishedDate", "Date inconnue")
    categories = ", ".join(volume_info.get("categories", ["Genre inconnu"]))
    description = volume_info.get("description", "Pas de description")
    page_count = volume_info.get("pageCount", "Inconnu")
    language = volume_info.get("language", "Langue inconnue")
    isbn_10 = isbn_13 = "Non disponible"
    industry_ids = volume_info.get("industryIdentifiers", [])
    for identifier in industry_ids:
        if identifier["type"] == "ISBN_10":
            isbn_10 = identifier["identifier"]
        if identifier["type"] == "ISBN_13":
            isbn_13 = identifier["identifier"]
    preview_link = volume_info.get("previewLink", "Non disponible")
    info_link = volume_info.get("infoLink", "Non disponible")
    average_rating = volume_info.get("averageRating", "Non noté")
    ratings_count = volume_info.get("ratingsCount", 0)
    image_link = volume_info.get("imageLinks", {}).get("thumbnail", "Non disponible")
    saleability = sale_info.get("saleability", "NON_DISPONIBLE")
    price = sale_info.get("listPrice", {}).get("amount", "Non disponible")
    currency = sale_info.get("listPrice", {}).get("currencyCode", "Non disponible")

    # compiling the information into a row for writing in a database

    row = [
        title,
        authors,
        publisher,
        published_date,
        categories,
        description,
        page_count,
        language,
        isbn_10,
        isbn_13,
        preview_link,
        info_link,
        average_rating,
        ratings_count,
        image_link,
        saleability,
        price,
        currency
    ]

    return row

# external functions

def gbooks_scrapper(
    scrapping_type: str,            # either "subject" or "inpublisher"
    api_key: str,                   # private API key
    order_by: str,                  # either "newest" or "relevance"
    csv_name: str,                  # name of CSV file to be saved in raw_data/
    list_of_search_keys: list       # list of search keys
    ) -> None:
    '''
    Scraps books from GBooks. Saves the CSVs in ./raw_data/gbooks_scrapping
    '''

    # ✅ Counter of books retrieved

    books_counter = 0

    # ✅ Create an empty CSV file with pre-set column names

    columns_names = ["Search Key", "API Request Number"] + KEY_BOOK_INFO_LIST

    folder_path = os.path.join(".","raw_data","gbooks_scrapping")
    os.makedirs(folder_path, exist_ok=True) # creation of the path if it doesn't exist
    output_csv_url = os.path.join(folder_path,f"{csv_name}.csv")

    with open(output_csv_url, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns_names)

    # ✅ Iterative loop on each search key of the list provided

    number_of_requests = 10     # maximum 400 results on a specific search, i.e., 10 requests of 40 results
    max_results = 40

    for search_key in list_of_search_keys:

        print(f"📚 Looking for : {search_key}")

        for request in range(0, number_of_requests-1):
            start_index = request * max_results
            print(f"➡️ Request #{request} | startIndex={start_index}")
            params = {
                'q': f'{scrapping_type}:{search_key}',
                'key': api_key,
                'orderBy': order_by,
                'printType': "books",
                'langRestrict': "fr",
                'maxResults': max_results,
                'startIndex': start_index
            }

            response = requests.get('https://www.googleapis.com/books/v1/volumes', params=params)
            data = response.json()

            if response.status_code != 200:
                print(f"❌ Error {response.status_code} for the request #{request + 1}")
                print(response.text)
                time.sleep(1)
                continue

            if 'items' not in data:
                print(f"⚠️ No result for the request #{request + 1}")
                time.sleep(1)
                continue

            books = []

            for book in data.get("items", []):
                row = [search_key,request] + database_row_inputer(book)
                books.append(row)

            # ✅ Appending the results of the request in the CSV file

            with open(output_csv_url, 'a', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(books)

            books_counter += len(books)
            print(f"✅ {len(books)} books added for Request #{request} | startIndex={start_index}")
            time.sleep(0.1)

    print(f"🎉 Scraping terminé ! {books_counter} livres consolidés et enregistrés dans {output_csv_url}")
    pass

def gbooks_lookup(search_words: list[str], show_output: bool = False) -> dict:
    '''
    Returns a dict with all the key book info, based on a list of words to be searched in Google Books' API.
    Search is set by default to French only.
    '''
    params = {
        'q': " ".join(search_words),
        'orderBy': "relevance",
        'printType': "books",
        'langRestrict': "fr",
        'maxResults': 1
    }
    response = requests.get('https://www.googleapis.com/books/v1/volumes', params=params)
    data = response.json()

    # retrieving the information

    try:
        book = data.get("items", [])[0]
        row = database_row_inputer(book)
        result = dict(zip(KEY_BOOK_INFO_LIST,row))
        if show_output == True:
            print(f"Search words: {search_words}")
            print("✅ A close match has been found on Google Books:")
            print(result)
        return result
    except:
        if show_output == True:
            print(f"Search words: {search_words}")
            print("❌ Error: no close match was found on Google Books")
        return None

def gbooks_lookup_multibooks(list_of_search_words: list[list[str]], show_output: bool = False) -> list[dict]:
    results = []
    counter = 0
    len_of_input = len(list_of_search_words)
    for search_words in list_of_search_words:
        if show_output == 1:
            counter += 1
            print(f"➡️ Searching for the book #{counter} out of {len_of_input}")
        results.append(gbooks_lookup(search_words,show_output))
    return results

def gbooks_look_isbn(isbn_input: str) -> dict:
    '''
    Returns a dict with all the key book info, based on an ISBN to be searched in Google Books' API.
    '''
    params = {
        'q': f'isbn:{isbn_input}',
        'maxResults': 1
    }
    response = requests.get('https://www.googleapis.com/books/v1/volumes', params=params)
    data = response.json()

    # retrieving the information

    try:
        book = data.get("items", [])[0]
        row = database_row_inputer(book)
        return dict(zip(KEY_BOOK_INFO_LIST,row))

    except (IndexError, KeyError):
        print(f"\n⚠️ Google Books API: ISBN {isbn_input} not found. Looking in local database...")

    # Recherche dans CSV
    try:
        df = pd.read_csv('raw_data/VF_data_base_consolidate_clean.csv', dtype=str)
        result = df[(df['ISBN-10'] == isbn_input) | (df['ISBN-13'] == isbn_input)]

        if not result.empty:
            print(f"\n✅ ISBN {isbn_input} found in local database.")
            row = result.iloc[0].to_dict()

            formatted_row = {
                'Title': row.get('Title'),
                'Authors': row.get('Authors'),
                'Publisher': row.get('Publisher'),
                'Published Date': row.get('Published Date'),
                'Categories': row.get('Categories'),
                'Description': row.get('Description'),
                'Page Count': int(row['Page Count']) if row.get('Page Count') and row['Page Count'].isdigit() else 0,
                'Language': row.get('Language'),
                'ISBN-10': row.get('ISBN-10'),
                'ISBN-13': row.get('ISBN-13'),
                'Preview Link': row.get('Preview Link'),
                'Info Link': row.get('Info Link'),
                'Average Rating': row.get('Average Rating'),
                'Ratings Count': int(row['Ratings Count']) if row.get('Ratings Count') and row['Ratings Count'].isdigit() else 0,
                'Image Link': row.get('Image Link'),
                'Saleability': row.get('Saleability'),
                'Price': float(row['Price']) if row.get('Price') not in ['Non disponible', '', None] else 'Non disponible',
                'Currency': row.get('Currency')
            }

            return formatted_row
        else:
            print(f"\n❌ ISBN {isbn_input} not found in local database.")
            return None

    except Exception as e:
        print(f"\n❌ Error reading local CSV: {e}")
        return None

if __name__ == "__main__":

    print(gbooks_look_isbn("2747306747"))
