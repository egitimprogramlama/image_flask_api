from cifar10_client import CIFAR10Client

def main():
    # Initialize client
    client = CIFAR10Client("http://16.171.55.173:5000")
    
    # Single prediction
    result = client.predict_single("cat.jpg")
    print(f"\nSingle prediction:")
    print(f"Class: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
    
    # Top 3 predictions
    result = client.predict_top3("dog.jpg")
    print(f"\nTop 3 predictions:")
    for pred, conf in zip(result['predictions'], result['confidences']):
        print(f"{pred}: {conf}%")
    
    # Batch prediction
    results = client.predict_batch(["cat.jpg", "dog.jpg", "bird.jpg"])
    print(f"\nBatch predictions:")
    for item in results['results']:
        print(f"\nFile: {item['filename']}")
        for pred, conf in zip(item['predictions'], item['confidences']):
            print(f"{pred}: {conf}%")

if __name__ == "__main__":
    main()