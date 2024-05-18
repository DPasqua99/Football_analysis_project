class TeamAssigner:
    def __init__(self) -> None:
        pass

    def get_clustering_model():
        pass

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        top_half_image = image[0: int(image.shape[0]/2), :]

        # get the clustering model
        kmeans = self.get_clustering_model(top_half_image)
        
    def assign_team_color(self, frame, player_detections):
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)