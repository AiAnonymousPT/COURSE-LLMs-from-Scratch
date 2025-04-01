import requests
import os

def download_gutenberg_books(num_books=10, output_dir="data/raw"):
    """
    Downloads the text of the top N books from Project Gutenberg and saves them as .txt files.

    Args:
        num_books (int): The number of top books to download.
        output_dir (str): The directory to save the downloaded books.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for book_id in range(1, num_books + 1):
        url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        output_file = os.path.join(output_dir, f"book_{book_id}.txt")

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

            if "text/plain" in response.headers.get('Content-Type', ''):
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"Downloaded book {book_id} and saved to {output_file}")
            else:
                print(f"Book {book_id} is not available in plain text format at {url}. Content-Type: {response.headers.get('Content-Type', 'N/A')}")


        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"Book {book_id} not found at {url} (404 Error).")
            else:
                print(f"HTTP Error downloading book {book_id} from {url}: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading book {book_id} from {url}: {e}")

if __name__ == "__main__":
    download_gutenberg_books()