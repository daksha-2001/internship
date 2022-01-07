from cassandra.cluster import Cluster
import pandas as pd
import logger

class cassandra_data:
    def __init__(self,file):
        self.log_path=file
        self.log_writer=logger.App_Logger()
    def cluster(self):
        try:
            cluster=Cluster()
            session_offline=cluster.connect()

            df = pd.DataFrame(list(session_offline.execute("select * from adult.adult;")))
            return df
        except Exception as e:
            self.log_writer.log(self.log_path, e)



