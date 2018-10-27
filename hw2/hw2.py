from feature_extractor import GridColorMomentsExtractor

dataset_folder = 'dataset/business_cards'
feature_extactor = GridColorMomentsExtractor(dataset_folder)
top1result, top5result, total_count = feature_extactor.extract_from_dataset()

print(f"top1:{top1result/total_count}")
print(f"top5:{top5result/total_count}")
