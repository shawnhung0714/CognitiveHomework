from feature_extractor import GridColorMomentsExtractor, GaborExtractor

if __name__ == '__main__':
    dataset_folder = 'dataset/dvd_covers'
    feature_extactor = GaborExtractor(dataset_folder)
    top1result, top5result, total_count = feature_extactor.extract_from_dataset()

    print(f"top1:{top1result/total_count}")
    print(f"top5:{top5result/total_count}")
