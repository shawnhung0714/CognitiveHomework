import feature_extractor
import os

if __name__ == '__main__':    
    for group in feature_extractor.all_group_list:
        feature_extactor = feature_extractor.GaborExtractor(group)
        top1result, top5result, total_count = feature_extactor.extract_from_dataset()
        print(f"top1:{top1result/total_count}")
        print(f"top5:{top5result/total_count}")

    feature_extactor.clean_up()
