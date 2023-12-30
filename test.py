from surprise.model_selection import train_test_split
from surprise import Dataset, Reader
import pandas as pd
# 영화 평가 점수 분포 탐색용
import matplotlib.pyplot as plt
# 모델트레이닝
from surprise import KNNBasic
from surprise import accuracy

rating_url = './data/ratings.dat'
rating_df = pd.io.parsers.read_csv(rating_url, names=[
                                   'user_id', 'movie_id', 'rating', 'time'], delimiter='::', engine='python')
movie_url = './data/movies.dat'
movie_df = pd.io.parsers.read_csv(movie_url, names=[
                                  'movie_id', 'title', 'genre'], delimiter='::', engine='python', encoding='ISO-8859-1')

# 데이터 확인
# print(rating_df)
# print(movie_df)
# print(len(rating_df['user_id'].unique()))
# print(len(rating_df['movie_id'].unique()))
# print(rating_df['rating'].hist())

# 트레인 셋, 데이터셋 분리
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    rating_df[['user_id', 'movie_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25)

algo = KNNBasic(k=40, min_k=1, sim_options={
                "user_based": True, "name": "cosine"})
algo.fit(trainset)
predictions = algo.test(testset)

# 모델 평가 RMSE
accuracy.rmse(predictions)
predictions = algo.test(testset[:20])
for _, iid, r_ui, predicted_rating, _ in predictions:
    print("Item id", iid, "|", "real rating :", r_ui,
          "|", "predicted rating :", predicted_rating)

# 4점 이상을 준 test 시청리스트 추출
user_watch_dict_list_test = test_df[test_df['rating']>=4].groupby('user_id')[['user_id', 'movie_id']].apply(lambda x: x['movie_id'].tolist())
user_metric = []

# 유저별 k개의 선호 리스트 추출
k = 3
for user in estimated_unwatched_dict:
  estimated_list = estimated_unwatched_dict[user].copy()
  estimated_list.sort(key=lambda tup: tup[1], reverse=True)
  try:
    top_k_prefer_list = [movie[0] for movie in estimated_list[:k]]
    actual_watch_list = user_watch_dict_list_test[int(user)]
    user_metric.append((user, top_k_prefer_list, actual_watch_list))
  except:
    print("list index out of range, exclude user " + str(user))

# 유저 한 명의 Precision
predictive_values = user_metric[0][1]
actual_values = set(user_metric[0][2])
tp = [pv for pv in predictive_values if pv in actual_values]
len(tp) / len(predictive_values)


def get_map(user_list):
  precision_list = []
  for user in user_list:
    predictive_values = user[1]
    actual_values = set(user[2])
    tp = [pv for pv in predictive_values if pv in actual_values]
    precision = len(tp) / len(predictive_values)
    precision_list.append(precision)
  return sum(precision_list) / len(precision_list)

get_map(user_metric)


def get_map_topk(k):
  user_metric = []
  for user in estimated_unwatched_dict:
    estimated_list = estimated_unwatched_dict[user].copy()
    estimated_list.sort(key=lambda tup: tup[1], reverse=True)
    try:
      top_k_prefer_list = [movie[0] for movie in estimated_list[:k]]
      actual_watch_list = user_watch_dict_list_test[user_watch_dict_list_test.index==user].values.tolist()[0]
      user_metric.append((user, top_k_prefer_list, actual_watch_list))
    except:
      pass
  
  precision_list = []
  for user in user_metric:
    predictive_values = user[1]
    actual_values = set(user[2])
    tp = [pv for pv in predictive_values if pv in actual_values]
    precision = len(tp) / len(predictive_values)
    precision_list.append(precision)
  return sum(precision_list) / len(precision_list)


k_param_list = range(1,30)
map_list = []
for k in k_param_list:    
  map_list.append(get_map_topk(k))


plt.plot(k_param_list, map_list)
plt.title('MAP by top k recommendation')
plt.ylabel('MAP', fontsize=12)
plt.xlabel('k', fontsize=12)
plt.show()