from IPython.display import HTML
class report:
    def init():
        return
    
    @staticmethod
    def report_nsl(df):
        html = df.to_html()
        # write html to file
        text_file = open("FYP/NSLKDD.html", "w")
        text_file.write(html)
        text_file.close()
    def report_unsw(df):
        html = df.to_html()
        # write html to file
        text_file = open("FYP/UNSW.html", "w")
        text_file.write(html)
        text_file.close()