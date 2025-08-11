# Auto-Modeler 🧠🔍  
Automatically find the **best classification model** and its **best hyperparameters** for any dataset.

---

## 🚀 What is Auto-Modeler?

Auto-Modeler is a simple Python-based program designed to make model selection and hyperparameter tuning easier for classification problems.

It:
- Tries multiple classification algorithms on your dataset
- Picks the one with the highest accuracy
- Then tunes its hyperparameters to find the best version of that model

✅ **It works on any classification dataset** — just provide the file path and the target column name.

---

## 📁 Project Structure

This project is built using **4 scripts**:

- `main.ipynb` – the notebook that runs everything  
- `preprocessing.py` – cleans the dataset (removes nulls, encodes labels, splits data)  
- `models.py` – scales datasets for certain algorithms, trains multiple classifiers and picks the best one  
- `hyperparameters_tuning.py` – tunes the selected model’s hyperparameters  

---

## ⚙️ How to Use

1. Clone this repo:
   git clone https://github.com/yourusername/auto-modeler.git
   cd auto-modeler
2. Install dependencies:
    pip install -r requirements.txt
    Open main.ipynb and set these variables:


3. Open `main.ipynb` and set these variables:
- `df_path = "path_to_your_dataset.csv"`
- `target_column = "your_target_column_name"`

4. Run all cells.

---

## ✅ What You’ll Get

- The best classification model for your dataset  
- Its best-performing hyperparameters  
- Evaluation metrics like accuracy, precision, and recall  

---

## 📊 Algorithms Used

Auto-Modeler evaluates the following classification algorithms:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (SVM)  

After selecting the top-performing model, it performs a **Grid Search** or **Randomized Search** to tune hyperparameters and boost performance.

---

## 💡 Why I Built This

Initially, I tried running all models with hyperparameter tuning in one go.  
It was slow, memory-intensive, and often crashed my system.

So I restructured the process to:
- First identify the best model  
- Then tune only that model’s parameters  

This results in faster execution, better resource management, and improved model performance.

---

## 🤝 Contribute

Have suggestions, improvements, or want to add more models?  
Feel free to fork the repo, open issues, or submit pull requests!

---

## ❓ Questions?

Got any questions or feedback?  
Feel free to reach out on [LinkedIn](https://www.linkedin.com/) or open an issue here on GitHub.

---

## 📄 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share it — just give credit where it’s due.



