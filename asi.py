import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    mo.md('# Book Recomendation Hackathon')
    return


@app.cell
def _():
    import os
    import warnings 

    os.environ['OPENBLUS_NUM_THREADS'] = '1'
    warnings.filterwarnings('ignore')
    return


@app.cell
def _():
    import pandas as pd
    import datetime as dt
    import numpy as np

    interactions = pd.read_csv('data/interactions.csv')
    editions = pd.read_csv('data/editions.csv')
    users = pd.read_csv('data/users.csv')
    book_genres = pd.read_csv('data/book_genres.csv')
    genres = pd.read_csv('data/genres.csv')
    authors = pd.read_csv('data/authors.csv') 
    target_users = pd.read_csv('submit/targets.csv') 
    target_interactions = pd.read_csv('submit/candidates.csv')

    print('all data frames have been loaded successfully')
    return book_genres, editions, interactions, np, pd, users


@app.cell
def _(interactions, pd):
    interactions['event_ts'] = pd.to_datetime(interactions['event_ts'])

    split_date = pd.Timestamp('2025-03-12')

    feature_source = interactions.loc[interactions['event_ts'] < split_date]
    train = interactions.loc[interactions['event_ts'] > split_date]
    return feature_source, train


@app.cell
def _(book_genres, editions, feature_source, train, users):
    import pandas as pd

    # Ячейка подготовки данных
    def prepare_training_data(feature_source_df, book_genres_df, editions_df, users_df, train_raw_df):
        # 1. Работа с жанрами
        # Группируем жанры в строку, чтобы при мердже не раздувать количество строк
        bg_processed = (
            book_genres_df.groupby('book_id')['genre_id']
            .apply(lambda x: ' '.join(x.astype(str)))
            .reset_index()
        )
    
        # 2. Обогащение изданий (editions)
        enriched_editions = editions_df.merge(bg_processed, on='book_id')
    
        # Считаем продуктивность автора (сколько раз его ID встречается в базе изданий)
        author_counts = enriched_editions['author_id'].value_counts()
        enriched_editions['author_productivity'] = enriched_editions['author_id'].map(author_counts)
    
        # 3. Сборка feature_source (основное хранилище признаков)
        fs = feature_source_df.drop('event_ts', axis=1)
        fs = fs.merge(users_df, on='user_id')
        fs = fs.merge(enriched_editions, on='edition_id')
    
        # Удаляем избыточные колонки
        fs = fs.drop(['book_id', 'publisher_id'], axis=1)
    
        # 4. Генерация фичей на основе взаимодействий
        # Важно: считаем статистики ДО того, как разделим на user/book features
        fs['edition_popularity_score'] = fs['edition_id'].map(fs['edition_id'].value_counts())
        fs['reader_mean_age'] = fs.groupby('edition_id')['age'].transform('mean')
        fs['book_age'] = 2026 - fs['publication_year']
        fs['user_mean_rating'] = fs.groupby('user_id')['rating'].transform('mean')
        fs['book_mean_rating'] = fs.groupby('edition_id')['rating'].transform('mean')
    
        # Убираем рейтинг, так как в классификации (event_type) он нам не нужен как признак
        fs_no_rating = fs.drop('rating', axis=1)
    
        # 5. Разделение на признаки пользователей и книг (для Negative Sampling)
        user_cols = ['user_id', 'gender', 'age', 'user_mean_rating']
    
        # Извлекаем уникальные фичи юзеров
        u_features = fs_no_rating[user_cols].drop_duplicates().reset_index(drop=True)
    
        # Извлекаем уникальные фичи книг (исключая колонки юзеров и тип события)
        b_cols = [c for c in fs_no_rating.columns if c not in user_cols and c != 'event_type']
        b_features = fs_no_rating[b_cols].drop_duplicates().reset_index(drop=True)
    
        # 6. Формирование финального train_1
        # Соединяем исходный train с подготовленными признаками
        t1 = train_raw_df.drop(['event_ts', 'rating'], axis=1, errors='ignore')
        t1 = t1.merge(b_features, on='edition_id').merge(u_features, on='user_id')
    
        return t1, b_features

    # В Marimo вызов будет выглядеть так:
    # (Выходные переменные теперь доступны для функции Negative Sampling)
    train_1, book_features = prepare_training_data(
        feature_source, 
        book_genres, 
        editions, 
        users, 
        train
    )
    return book_features, pd, train_1


@app.cell
def _(np, pd):
    def create_random_negative_samples(
        train_df, 
        books_df, 
        negative_fraction=3, 
        seed=42
    ):
        """
        Генерирует негативные примеры (event_type=0) случайным образом.
        """
        # Фиксируем seed для воспроизводимости
        rng = np.random.default_rng(seed)

        needed_cols = [
            'user_id', 'edition_id', 'event_type', 'gender', 'age', 'author_id', 
            'publication_year', 'age_restriction', 'language_id', 'title', 
            'description', 'genre_id', 'author_productivity', 
            'edition_popularity_score', 'reader_mean_age', 'book_age', 
            'user_mean_rating', 'book_mean_rating'
        ]

        # 1. Считаем, сколько сэмплов нужно каждому юзеру
        num_samples = train_df['user_id'].value_counts() * negative_fraction
        max_books = len(books_df)
        num_samples = num_samples.clip(upper=max_books)

        # 2. Подготовка юзеров
        # Берем уникальные характеристики пользователей для джойна
        _user_cols = ['user_id', 'gender', 'age', 'user_mean_rating']
        _user_features = train_df[_user_cols].drop_duplicates(subset=['user_id']).set_index('user_id')

        # 3. Массивы для генерации
        user_ids = num_samples.index.to_numpy()
        counts = num_samples.to_numpy()

        # Повторяем ID юзеров
        repeated_user_ids = np.repeat(user_ids, counts)
        total_samples = len(repeated_user_ids)

        # 4. Рандомный выбор книг
        # Генерируем случайные индексы
        random_indices = rng.integers(0, max_books, size=total_samples)

        # Выбираем книги (сброс индекса важен для корректного iloc)
        sampled_books = books_df.reset_index(drop=True).iloc[random_indices].reset_index(drop=True)

        # 5. Собираем признаки юзеров
        sampled_users = _user_features.loc[repeated_user_ids].reset_index(drop=True)

        # 6. Сборка негативного датасета
        neg_df = pd.concat([sampled_users, sampled_books], axis=1)
        neg_df['user_id'] = repeated_user_ids
        neg_df['event_type'] = 0

        # Оставляем только нужные колонки (проверка на случай отсутствия колонок в book_features)
        # Если каких-то колонок нет, pandas выдаст ошибку, но предполагаем, что входные данные корректны.
        neg_df = neg_df[needed_cols]

        # 7. Объединение с трейном
        final_df = pd.concat([train_df, neg_df], ignore_index=True)

        # 8. Удаление коллизий (если рандом попал в реальную книгу)
        # Сортируем: event_type=1 (реальные) выше, чем 0
        final_df = final_df.sort_values(by='event_type', ascending=False)
        # Удаляем дубли, оставляя первую запись (реальную)
        final_df = final_df.drop_duplicates(subset=['user_id', 'edition_id'], keep='first')

        final_df['edition_id'] = final_df['edition_id'].astype(str)

        return final_df.reset_index(drop=True)

    return (create_random_negative_samples,)


@app.cell
def _(book_features, create_random_negative_samples, train_1):
    train_2 = create_random_negative_samples(
        train_df=train_1, 
        books_df=book_features, 
        negative_fraction=3
    )

    # Проверка результата (вывод заголовка)
    train_2.head()
    return (train_2,)


@app.cell
def _(train_2):
    train_2.head(3)
    return


@app.cell
def _(
    Pool,
    cat_features,
    label_train,
    text_features,
    train_2,
    train_test_split,
):
    # 1. Сначала готовим ЕДИНЫЙ датасет с правильным порядком
    # Сбрасываем индекс СРАЗУ, чтобы он был 0, 1, 2... везде
    train_3 = train_2.sort_values('user_id').reset_index(drop=True)
    queries = train_3['user_id']
    # 2. Выделяем группы (Queries) и таргет (Label) ПЕРЕД удалением колонок
    label = train_3['event_type']
    drop_cols = ['event_type', 'edition_id', 'user_id', 'edition_popularity_score', 'user_mean_rating', 'book_mean_rating', 'reader_mean_age']
    X = train_3.drop(columns=drop_cols)
    # 3. Готовим признаки (X) - удаляем ВСЁ лишнее тут
    # ВАЖНО: Убираем user_id отсюда, чтобы не было лика!
    cat_features_safe = ['author_id', 'language_id', 'gender', 'genre_id']
    for col in cat_features_safe:
        X[col] = X[col].astype(str).replace('nan', 'unknown')
    X_train, X_test, y_train, y_test, q_train, q_test = train_test_split(X, label, queries, test_size=0.33, random_state=42, shuffle=False)
    print(f'Размер групп в трейне: {q_train.value_counts().mean():.2f} книг на юзера')
    # 4. Приводим категории к строкам
    text_cols = ['title', 'description', 'genre_id']
    for col in text_cols:
        X_train[col] = X_train[col].fillna('none').astype(str)
        X_test[col] = X_test[col].fillna('none').astype(str)
    # 5. МАГИЧЕСКИЙ СПЛИТ: Передаем X, y и queries ВМЕСТЕ
    # shuffle=False обязателен для ранкера, чтобы не разорвать группы юзеров
    train_pool = Pool(data=X_train, label=label_train, group_id=q_train, cat_features=cat_features, text_features=text_features)
    # 6. Проверка на вшивость (Обязательно посмотри этот принт!)
    # Если тут будет 1.0 — значит данные кривые. Должно быть 5-10-20.
    # Заполни пустоты во ВСЕХ текстовых признаках
    # Теперь создавай Pool
    val_pool = Pool(data=X_test, label=y_test, group_id=q_test, cat_features=cat_features_safe, text_features=text_cols)  # проверь список по своему коду
    return train_pool, val_pool


@app.cell
def _(train_pool, val_pool):
    # magic command not supported in marimo; please file an issue to add support
    # %%time

    from catboost import CatBoostRanker

    TASK_TYPE = 'CPU'

    model = CatBoostRanker(
        iterations=1000, 
        learning_rate=0.1, 
        loss_function='YetiRank', 
        eval_metric='NDCG:top=20', 
        random_seed=42, 
        task_type=TASK_TYPE, 
        metric_period=100, 
        use_best_model=True, 
        early_stopping_rounds=100
    )

    model.fit(
        train_pool,
        eval_set=val_pool, 
        plot=True
    )
    return (model,)


@app.cell
def _(model, pd, train_pool):
    # 1. Проверяем, что модель обучена. Если выведет True, значит всё ок.
    print(f"Is model trained: {model.is_fitted()}")

    # 2. Явно запрашиваем важность
    importances = model.get_feature_importance(train_pool)
    feature_names = model.feature_names_

    # 3. Собираем таблицу
    fea_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print(fea_imp)
    return


if __name__ == "__main__":
    app.run()
