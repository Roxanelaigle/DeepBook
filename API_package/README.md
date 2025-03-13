# README

## Description
This module creates the final API, which will use the `full_model()` function (still under development).

## `full_model()` function

- **Inputs**:
  - `image_array` **or** `isbn` *(string)*
  - `curiosity_level` *(integer)*

- **Output**:
  - A JSON response (Python dictionary)

### Output JSON format

```json
{
  "input_book": {
    "title": "xx",
    "authors": "xx",
    "image_link": "image_url",
    "isbn": "isbn_13",
    "description": "description"
  },
  "output_books": [
    {
      "title": "xx",
      "authors": "xx",
      "image_link": "image_url",
      "isbn": "isbn_13",
      "description": "description"
    }
  ]
}

