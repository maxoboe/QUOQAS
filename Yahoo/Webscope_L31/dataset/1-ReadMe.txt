Dataset: ydata-yanswers-inf-conv-questions-v1_0
 
Questions on Yahoo Answers labeled as either informational or conversational, version 1.0
 
=====================================================================
This dataset is provided as part of the Yahoo! Webscope program, to be
used for approved non-commercial research purposes by recipients who
have signed a Data Sharing Agreement with Yahoo!. This dataset is not
to be redistributed. No personally identifying information is available
in this dataset. More information about the Yahoo! Webscope program is
available at http://research.yahoo.com
=====================================================================
 
Full description:
This dataset consists of over 4000 questions from the Yahoo Answers community question and answering website, labeled as informational or conversational. 
The dataset consists of two files: 
batch1.csv
batch2.csv
Batch1 contains 1088 labeled questions based on the input of two experts, while batch2 contains 2966 labeled questions based on the input of undergraduate students, who were instructed and graded for the task. 
In both batches, each of the questions was labeled by two annotators. In each of the batches, each question is represented as a line with the following fields (fields are comma-separated): 
1). Title - the title of the question in Yahoo Answers.
2). Description - the description of the question in Yahoo Answers (if exists, otherwise the string "NULL"). 
3). Category - the direct category of the question as assigned on Yahoo Answers. 
4). Top - the high-level category of the question as assigned on Yahoo Answers. 
5). URL - the URL of the Yahoo Answers page. In addition to the information directly included in this dataset, the Yahoo Answers page typically presents extra information, such as the question's answers, votes on answers, links to the profile pages of the asker and answerers, and more.
6). Label - '0' for informational; '1' for conversational; '2' for borderline (the two annotators disagreed on the label). 


===
Note that public data can change over time. In case you use the link to download more data from the Yahoo Answers page, you must make sure you do not publish Yahoo Answer's user data without first confirming that the information is still public
===

Spam questions and answers are not included.
We relied on the Yahoo Answers moderation process to filter out offensive questions.

Data examples: 
http://answers.yahoo.com/question/index?qid=20070811012412AADS4aU,mount Shasta?,I'm looking for places to go at mount shasta? ;please answer if you know my family's going tomorrow,Travel,Sacramento,0
http://answers.yahoo.com/question/index?qid=20100728211205AAOOoYr,what's the absolute best thing you've ever eaten at a 1st class restaurant?,Was it worth the money?,Dining Out,Other - Dining Out,1
http://answers.yahoo.com/question/index?qid=20070424132042AAvL94u,Is it ok to wear a polo and jeans to a club/bar?,NULL,Beauty & Style,Fashion & Accessories,2
http://answers.yahoo.com/question/index?qid=20130326101300AAuPN6g,what is the differences and similarities MRI and CT?,NULL,Science & Mathematics,Medicine,0
http://answers.yahoo.com/question/index?qid=20100517162425AAa5jwB,Nobody seems to care about my birthday :'|?,It probably sounds stupid to you but I can see how it is all going to pan out with me feeling unwanted isolated and uncared about and bam when I'm in tat frame of mind death is my only option. You have no idea what it's like to be depressed,Family & Relationships,Family,1