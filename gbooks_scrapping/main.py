import time
import csv
import pandas as pd
import requests
import os

def gbooks_scrapper(
    scrapping_type: str,            # either "subject" or "inpublisher"
    api_key: str,                   # private API key
    order_by: str,                  # either "newest" or "relevance"
    csv_name: str,                  # name of CSV file to be saved in raw_data/
    list_of_search_keys: list       # list of search keys
    ) -> None:

    # ✅ Counter of books retrieved

    books_counter = 0

    # ✅ Create an empty CSV file with pre-set column names

    columns_names = [
            "Search Key", "API Request Number", "Title", "Authors", "Publisher", "Published Date", "Categories",
            "Description", "Page Count", "Language", "ISBN-10", "ISBN-13", "Preview Link", "Info Link",
            "Average Rating", "Ratings Count", "Image Link", "Saleability", "Price", "Currency"
        ]

    folder_path = os.path.join(".","raw_data","gbooks_scrapping_data")
    os.makedirs(folder_path, exist_ok=True) # creation of the path if it doesn't exist
    output_csv_url = os.path.join(folder_path,f"{csv_name}.csv")

    with open(output_csv_url, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns_names)

    # ✅ Iterative loop on each search key of the list provided

    for search_key in list_of_search_keys:

        print(f"📚 Looking for : {search_key}")

        for request in range(0, 9): # maximum 400 results on a search, i.e., 10 pages of 40 results
            start_index = request * 40 # maximum 40 results in each API call
            print(f"➡️ Request #{request} | startIndex={start_index}")
            params = {
                'q': f'{scrapping_type}:{search_key}',
                'key': api_key,
                'orderBy': order_by,
                'printType': "books",
                'langRestrict': "fr",
                'maxResults': 40, # maximum 40 results in each API call
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

                # ✅ Ligne unique pour le CSV et le DataFrame
                row = [
                    search_key,
                    request,
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
