import feature_extractor
import os

if __name__ == '__main__':
    group_list = [
        'dvd_covers',
        'cd_covers',
        'book_covers',
        'museum_paintings',
        'video_frames',
        'business_cards',
    ]
    for group in group_list:
        dataset_folder = os.path.join('dataset', group)
        feature_extactor = feature_extractor.GaborExtractor(dataset_folder)
        top1result, top5result, total_count = feature_extactor.extract_from_dataset()
        print(f"top1:{top1result/total_count}")
        print(f"top5:{top5result/total_count}")
        feature_extactor.clean_up()
