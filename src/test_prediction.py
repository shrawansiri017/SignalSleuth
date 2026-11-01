from predict_news import predict_news

sample_news = """
The government announced a new economic policy
to improve employment opportunities across the country.
"""

result = predict_news(sample_news)

print("SIGNALSLEUTH - NEWS ANALYSIS")
print("----------------------------")

for key, value in result.items():
    print(f"{key}: {value}")