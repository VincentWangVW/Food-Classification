import sys
import re
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------
# 1)preprocessing functions
# ------------------------------------------------------------------------

def number_to_words(n):
    """
    Convert an integer 1–100 to its English words representation.
    """
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen",
             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty",
            "fifty", "sixty", "seventy", "eighty", "ninety"]
    if 1 <= n < 10:
        return ones[n]
    elif 10 <= n < 20:
        return teens[n - 10]
    elif 20 <= n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 != 0 else "")
    elif n == 100:
        return "one hundred"
    return str(n)

def preprocess_ingredient_count(text):
    """
    Convert ingredient count responses to numeric values.
    (Same logic you used in training.)
    """
    text = str(text).strip().lower()
    
    # Build a dictionary for spelled-out numbers from 1 to 100.
    word_number_dict = {number_to_words(i): i for i in range(1, 101)}

    # Replace spelled-out numbers with digits
    for spelled, numeric in word_number_dict.items():
        # Replace hyphenated forms,
        text = re.sub(r'\b' + re.escape(spelled) + r'\b', str(numeric), text)
        # Replace a space-separated variant
        if '-' in spelled:
            no_hyphen = spelled.replace('-', ' ')
            text = re.sub(r'\b' + re.escape(no_hyphen) + r'\b', str(numeric), text)

    # Handle ranges with "-" or "to"
    if '-' in text or 'to' in text:
        numbers = [float(n) for n in re.findall(r'\d+\.?\d*', text)]
        if numbers:
            return np.mean(numbers)

    # Handle "or"
    if 'or' in text:
        numbers = [float(n) for n in re.findall(r'\d+\.?\d*', text)]
        if numbers and len(numbers) >= 2:
            return np.mean(numbers)

    # If digits are present, return the first found
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        return float(numbers[0])

    # If commas, treat as list
    if ',' in text:
        items = [item.strip() for item in text.split(',') if item.strip()]
        if items:
            return len(items)

    # If newlines, treat as list
    if "\n" in text:
        items = [line.strip() for line in text.split("\n") if line.strip()]
        if len(items) > 1:
            return len(items)
        elif len(items) == 1:
            return 1

    # If "and" or "or"
    if " and " in text:
        items = [x.strip() for x in text.split(" and ") if x.strip()]
        if len(items) > 1:
            return len(items)
    if " or " in text:
        items = [x.strip() for x in text.split(" or ") if x.strip()]
        if len(items) > 1:
            return len(items)

    # If anything left, assume 1
    if text:
        return np.nan
    return np.nan

def pre_process_1(df: pd.DataFrame) -> pd.DataFrame:
    return df

def pre_process_2(df: pd.DataFrame) -> pd.DataFrame:
    df['2. ingredient count'] = df['2. ingredient count'].apply(preprocess_ingredient_count)
    median_val = df['2. ingredient count'].median()
    df['2. ingredient count'] = df['2. ingredient count'].fillna(median_val).round().astype(int)
    return df

def pre_process_3(df: pd.DataFrame) -> pd.DataFrame:
    options = ["Week day lunch",
               "Week day dinner",
               "Weekend lunch",
               "Weekend dinner",
               "At a party",
               "Late night snack"]
    for option in options:
        df[option] = df["3. setting"].astype(str).apply(lambda x: 1 if option in x else 0)
    df.drop(columns=["3. setting"], inplace=True)
    return df



def pre_process_4(df: pd.DataFrame) -> pd.DataFrame:
    no_dollar_int = r'\$?\d+(\.\d{1,2})?\$?'
    range_pattern = r'\$?\d+(\.\d{1,2})?\s*[-~]\s*\$?\d+(\.\d{1,2})?'

    for index, value in df['4. price'].items():
        value_str = str(value).strip()
        a = re.match(no_dollar_int, value_str)
        b = re.match(range_pattern, value_str)
        c = re.search(range_pattern, value_str)
        d = re.search(no_dollar_int, value_str)

        # Also check spelled-out numbers
        word_number_dict = {number_to_words(i): i for i in range(1, 101)}
        e = [word_number_dict[word.lower()] 
             for word in value_str.split() 
             if word.lower() in word_number_dict]

        if b:
            # A direct range match
            range_values = b.group().replace('~', '-').split('-')
            vals = []
            for rv in range_values:
                cleaned = ''.join(ch for ch in rv if ch.isdigit() or ch == '.')
                vals.append(float(cleaned))
            avg_val = round(sum(vals) / len(vals), 2)
            df.at[index, '4. price'] = avg_val
        elif a:
            # Single numeric
            cleaned_a = ''.join(ch for ch in a.group() if ch.isdigit() or ch == '.')
            df.at[index, '4. price'] = float(cleaned_a)
        elif c:
            # Range with search
            range_values = c.group().replace('~', '-').split('-')
            vals = []
            for rv in range_values:
                cleaned = ''.join(ch for ch in rv if ch.isdigit() or ch == '.')
                vals.append(float(cleaned))
            avg_val = round(sum(vals) / len(vals), 2)
            df.at[index, '4. price'] = avg_val
        elif d:
            cleaned_d = ''.join(ch for ch in d.group() if ch.isdigit() or ch == '.')
            df.at[index, '4. price'] = float(cleaned_d)
        elif e:
            df.at[index, '4. price'] = float(e[0])
        else:
            df.at[index, '4. price'] = np.nan

    df['4. price'] = pd.to_numeric(df['4. price'], errors='coerce')
    median_val = df['4. price'].median()
    df['4. price'] = df['4. price'].fillna(median_val).round(2).astype(float)
    return df

# def pre_process_5(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
#     # Clean the movie column
#     df["5. movie"] = df["5. movie"].astype(str).str.lower().str.strip()

#     # Find the top_n frequent movies
#     counts = df["5. movie"].value_counts()
#     top_movies = counts.nlargest(top_n).index.tolist()

#     # Map responses
#     df["5. movie_mapped"] = df["5. movie"].apply(lambda x: x if x in top_movies else "other")

#     # One-hot encode
#     dummies = pd.get_dummies(df["5. movie_mapped"], prefix="movie", dtype=int)
#     for col in dummies.columns:
#         df[col] = dummies[col]

#     df.drop(columns=["5. movie", "5. movie_mapped"], inplace=True)
#     return df
movies = ["movie_aladdin",
          "movie_avengers",
          "movie_cloudy with a chance of meatballs",
          "movie_finding nemo",
          "movie_home alone",
          "movie_jiro dreams of sushi",
          "movie_kill bill",
          "movie_kung fu panda",
          "movie_nan",
          "movie_no movie",
          "movie_none",
          "movie_other",
          "movie_ratatouille",
          "movie_rush hour",
          "movie_spiderman",
          "movie_spirited away",
          "movie_teenage mutant ninja turtles",
          "movie_the avengers",
          "movie_the dictator",
          "movie_the godfather",
          "movie_your name"]
def pre_process_5(df: pd.DataFrame, top_movies: list[str]) -> pd.DataFrame:
    """
    Preprocesses the test set for Q5 using the top movie list from the training data.
    
    This version:
      - Cleans the movie column.
      - Maps to one-hot columns (all numeric).
      - Creates a numeric 'predicted_movie_index' column.
      - Drops original string columns so that the final output is entirely numeric.
    """
    # 1) Clean the movie column to ensure consistent strings
    df["5. movie"] = df["5. movie"].astype(str).str.lower().str.strip()

    # 2) Create a list of allowed movie names (no 'movie_' prefix for matching)
    allowed_movies = [m.replace("movie_", "") for m in top_movies]

    # 3) Map the movie to an allowed name or "other"
    df["5. movie_mapped"] = df["5. movie"].apply(
        lambda x: x if x in allowed_movies else "other"
    )

    # 4) Create one-hot encoding (movie_ prefix to match the columns in top_movies)
    dummies = pd.get_dummies(df["5. movie_mapped"], prefix="movie", dtype=int)
    # Ensure all columns from top_movies are present
    dummies = dummies.reindex(columns=top_movies, fill_value=0)

    # 5) Concatenate the numeric dummy columns back into df
    df = pd.concat([df, dummies], axis=1)

    # 6) Create a numeric index for each mapped movie
    #    Example: if "spiderman" is the third item in allowed_movies, it gets index 2.
    #    "other" must be in allowed_movies to avoid KeyErrors if we want to index it too.
    # def get_movie_index(mapped_value: str) -> int:
    #     # If you're certain "other" is in allowed_movies, the below line is enough:
    #     return allowed_movies.index(mapped_value) if mapped_value in allowed_movies \
    #            else allowed_movies.index("other")

    # df["predicted_movie_index"] = df["5. movie_mapped"].apply(get_movie_index)

    # 7) Now drop the original text columns to keep only numeric columns
    df.drop(columns=["5. movie", "5. movie_mapped"], inplace=True)

    # 8) Return the final DataFrame, which should be entirely numeric
    return df

drink_categories = {
    "Water": ["water", "mineral water", "tap water", "cold water", "lemon water"],
    "Soft Drink": [
        "coke", "cola", "coca-cola", "pepsi", "sprite", "fanta", "crush",
        "dr pepper", "dr. pepper", "root beer", "ginger ale", "soda",
        "soft drink", "pop", "carbonated drink"
    ],
    "Juice": [
        "orange juice", "apple juice", "grape juice", "mango juice", "cranberry juice",
        "pineapple juice", "fruit juice", "lemonade"
    ],
    "Tea": [
        "tea", "iced tea", "ice tea", "green tea", "black tea", "oolong tea",
        "matcha", "barley tea", "jasmine tea", "hot tea"
    ],
    "Coffee": ["coffee", "espresso", "latte", "cappuccino"],
    "Milk": ["milk", "chocolate milk", "milkshake", "soy milk", "almond milk"],
    "Alcoholic": [
        "beer", "wine", "red wine", "white wine", "sake", "soju", "whiskey",
        "vodka", "cocktail", "rum", "martini", "baijiu", "champagne"
    ],
    "Fermented": ["kombucha", "lassi", "ayran", "yakult"],
    "Other": ["soup", "miso soup", "no drink", "none"]
}

def categorize_drink(response: str) -> list:
    response = response.lower()
    response = re.sub(r'[^\w\s]', '', response)
    matched_categories = set()
    for category, keywords in drink_categories.items():
        for keyword in keywords:
            if keyword in response:
                matched_categories.add(category)
                break
    if not matched_categories:
        matched_categories.add("Other")
    return list(matched_categories)

def pre_process_6(df: pd.DataFrame) -> pd.DataFrame:
    df["drink_categories"] = df["6. drink"].astype(str).apply(categorize_drink)

    for category in drink_categories.keys():
        col_name = f"drink_{category.lower().replace(' ', '_')}"
        df[col_name] = df["drink_categories"].apply(lambda cats: 1 if category in cats else 0)

    df.drop(columns=["6. drink", "drink_categories"], inplace=True)
    return df

def pre_process_7(df: pd.DataFrame) -> pd.DataFrame:
    options = ["Parents", "Siblings", "Friends", "Teachers", "Strangers"]
    for option in options:
        col_name = "remind_" + option.lower()
        df[col_name] = df["7. remind"].astype(str).apply(lambda x: 1 if option in x else 0)
    df.drop(columns=["7. remind"], inplace=True)
    return df

def pre_process_8(df: pd.DataFrame) -> pd.DataFrame:
    hot_sauce_mapping = {
        "None": 0,
        "A little (mild)": 1,
        "A moderate amount (medium)": 2,
        "A lot (hot)": 3,
        "I will have some of this food item with my hot sauce": 4
    }
    df["8. hot sauce"] = df["8. hot sauce"].astype(str).str.strip().apply(lambda x: hot_sauce_mapping.get(x, 0))
    return df

def pre_process_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the exact same 8 steps you used for training, then scales the columns:
    '1. complexity', '2. ingredient count', '4. price', and '8. hot sauce'.

    Returns:
      df (pd.DataFrame): The transformed DataFrame.
    """
    # Rename columns
    df.rename(columns={
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': '1. complexity',
        'Q2: How many ingredients would you expect this food item to contain?': '2. ingredient count',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': '3. setting',
        'Q4: How much would you expect to pay for one serving of this food item?': '4. price',
        'Q5: What movie do you think of when thinking of this food item?': '5. movie',
        'Q6: What drink would you pair with this food item?': '6. drink',
        'Q7: When you think about this food item, who does it remind you of?': '7. remind',
        'Q8: How much hot sauce would you add to this food item?': '8. hot sauce'
    }, inplace=True)
    
    df = pre_process_1(df)
    df = pre_process_2(df)
    df = pre_process_3(df)
    df = pre_process_4(df)
    df = pre_process_5(df,movies)
    df = pre_process_6(df)
    df = pre_process_7(df)
    df = pre_process_8(df)

    # Drop 'id' if it exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'], errors='ignore')

    # Scale the specified columns using basic z-score scaling.
    scale_cols = ["1. complexity", "2. ingredient count", "4. price", "8. hot sauce"]
    for col in scale_cols:
        mean_val = df[col].mean()
        std_val = df[col].std(ddof=0)
        df[col] = (df[col] - mean_val) / std_val

    return df



# ------------------------------------------------------------------------
# 2) MLPClassifierMulticlass
# ------------------------------------------------------------------------
class MLPClassifierMulticlass:
    """
    multiclass MLP for inference only.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        alpha,
        learning_rate_init,
        max_iter,
        random_state
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state

        # Placeholders for weights (to be loaded)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def load_weights(self, W1, b1, W2, b2):
        """Load pre-trained weights/biases."""
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def predict_proba(self, X):
        """Forward pass: returns class probability distributions."""
        Z1 = X.dot(self.W1) + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        exps = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def predict(self, X):
        """Return integer predictions."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)



def predict_all(csv_filename):
    # 1) Load model params
    data = np.load("my_model.npz", allow_pickle=True)
    W1 = data["W1"]
    b1 = data["b1"]
    W2 = data["W2"]
    b2 = data["b2"]
    
    input_dim = data["input_dim"].item()
    output_dim = data["output_dim"].item()
    hidden_dim = data["hidden_dim"].item()
    alpha = data["alpha"].item()
    learning_rate_init = data["learning_rate_init"].item()
    max_iter = data["max_iter"].item()
    random_state = data["random_state"].item()

    # 2) Hardcode label order: Pizza=0, Sushi=1, Shawarma=2
    forced_categories = ["Pizza", "Sushi", "Shawarma"]
    label_mapping = dict(enumerate(forced_categories))

    # 3) Create model with same hyperparams
    model = MLPClassifierMulticlass(
        input_dim, output_dim, hidden_dim,
        alpha, learning_rate_init,
        max_iter, random_state
    )
    model.load_weights(W1, b1, W2, b2)

    # 4) Read CSV
    df_raw = pd.read_csv(csv_filename)

    has_label = ("Label" in df_raw.columns)
    if has_label:
        df_raw["Label"] = pd.Categorical(
            df_raw["Label"],
            categories=forced_categories,
            ordered=True
        )

    # 5) Preprocess
    df_processed = pre_process_pipeline(df_raw.copy())

    if "Label" in df_processed.columns:
        true_labels_categorical = df_processed["Label"]
        df_processed = df_processed.drop(columns=["Label"])
    else:
        true_labels_categorical = None

    # 6) Convert to numpy
    X_test = df_processed.values
    if X_test.shape[1] != input_dim:
        print(f"Error: CSV has {X_test.shape[1]} features, but model expects {input_dim}.")
        # sys.exit(1)

    # 7) Predict
    y_pred_int = model.predict(X_test)
    mapped_preds = [label_mapping[i] for i in y_pred_int]

    # 8) If we have a true Label column, compute accuracy
    if has_label:
        true_labels_str = true_labels_categorical.astype(str).values
        preds_array = np.array(mapped_preds)
        accuracy = np.mean(true_labels_str == preds_array)
        return mapped_preds
    else:
        return mapped_preds

# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python pred.py <test_csv>")
#         sys.exit(1)

#     test_csv = sys.argv[1]
#     result = predict_all(test_csv)

#     if isinstance(result, tuple):
#         preds, acc = result
#         print("Prediction Accuracy:", acc)
#         for p in preds:
#             print(p)
#     else:
#         # Just predicted labels
#         for p in result:
#             print(p)
#     print("Prediction Accuracy:", acc)
# if __name__ == "__main__":
#     main()
