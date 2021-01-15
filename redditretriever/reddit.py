"""
Module to extract information from reddit subreddits and place them in a dataframe
"""
import praw
import pandas as pd

class RedditRetriever:
    """
    A class to retrieve and store information in a desired format from reddit
    """
    def __init__(self):
        reddit = praw.Reddit(client_id="",
                     client_secret="",
                     password="",
                     user_agent="judgment",
                     username="hessik")

        print(reddit.read_only)  # Output: False
        print(reddit.user.me())
        self.subreddit = reddit.subreddit("RoastMe")
        self.topics_dict = { "title":[],
                "score":[],
                "id":[],
                "url":[],
                "comms_num": [],
                "created": [],
                "body":[],
                "roast":[],
                "image":[]}
        print(self.subreddit.display_name)  # output: redditdev

    def get_submissions(self, num_entries):
        """
        Retrieving Submissions and updating the topic dictionary
        """
        for submission in self.subreddit.top(limit=num_entries):
            self.topics_dict["title"].append(submission.title)
            self.topics_dict["score"].append(submission.score)
            self.topics_dict["id"].append(submission.id)
            self.topics_dict["url"].append(submission.url)
            self.topics_dict["comms_num"].append(submission.num_comments)
            self.topics_dict["created"].append(submission.created)
            self.topics_dict["body"].append(submission.selftext)
            self.topics_dict["roast"].append(submission.comments[0].body)
            self.topics_dict["image"].append('"<img src="'+ submission.url + '" width="60" >"')

        return pd.DataFrame(self.topics_dict)




if __name__ == "__main__":
    red = RedditRetriever()
    topics_data = red.get_submissions(1000)
    print(topics_data)
    topics_data.to_csv('reddit.csv')
    topics_data.to_html('webpage.html',escape=False)
