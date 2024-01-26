from utils.transaction import Transaction
from collections import defaultdict
from utils.actor import Actor
import tabulate

class Community:
    last_assigned_id = 0
    
    def __init__(self, actors, sended, received, nb_transactions, nb_unique_transactions, transactions):
        """
        Initialize a Community object.

        Args:
        actors (list): List of actors in the community.
        sended (float): Total amount sent by the community.
        received (float): Total amount received by the community.
        nb_transactions (int): Total number of transactions made by the community.
        nb_unique_transactions (int): Total number of unique transactions made by the community.
        transactions (list): List of transactions made by the community.
        """
        Community.last_assigned_id += 1
        self.id = Community.last_assigned_id
        
        self.members = actors
        self.size = len(actors)
        
        self.sended = sended
        self.received = received
        self.volume = sended + received
        
        self.nb_transactions = nb_transactions
        self.nb_unique_transactions = nb_unique_transactions
        
        self.transactions = transactions
        self.transactions_volume_by_day = []
        
    def process_volume_by_day(self):
        """
        Get the transactions of this actor grouped by date.

        Returns:
        list: The transactions of this actor grouped by date.
        """
        transactions_by_date = defaultdict(list)  
        for transaction in self.transactions:
            transaction_date = transaction.get_date().date()
            transactions_by_date[transaction_date].append(transaction)
        
        actor = Actor("COMMUNITY_TOTAL_VOLUME", "-1")
        for transaction_date, transactions in transactions_by_date.items():
            volume_day = sum([transaction.get_value() for transaction in transactions])
            nb_transactions_day = sum([transaction.get_nb_transactions() for transaction in transactions])
            
            transaction = Transaction(self, actor, volume_day, nb_transactions_day, transaction_date)
            self.transactions_volume_by_day.append(transaction)   
         
        self.transactions_volume_by_day = sorted(self.transactions_volume_by_day, key=lambda x: x.get_date())
        return self.transactions_volume_by_day
    
    def get_transactions_volume_by_day(self):
        """
        Get the transactions of this actor grouped by date.

        Returns:
        list: The transactions of this actor grouped by date.
        """
        return self.transactions_volume_by_day
    
    def get_volume(self):
        """
        Get the total volume (sent + received) of the community.

        Returns:
        float: The total volume of the community.
        """
        return self.volume
    
    def get_sended(self):
        """
        Get the total amount sent by the community.

        Returns:
        float: The total amount sent by the community.
        """
        return self.sended
    
    def get_received(self):
        """
        Get the total amount received by the community.

        Returns:
        float: The total amount received by the community.
        """
        return self.received
    
    def get_nb_transactions(self):
        """
        Get the total number of transactions made by the community.

        Returns:
        int: The total number of transactions made by the community.
        """
        return self.nb_transactions
    
    def get_nb_unique_transactions(self):
        """
        Get the total number of unique transactions made by the community.

        Returns:
        int: The total number of unique transactions made by the community.
        """
        return self.nb_unique_transactions
    
    def get_received(self):
        """
        Get the total amount received by the community.

        Returns:
        float: The total amount received by the community.
        """
        return self.received

    def get_size(self):
        """
        Get the size (number of members) of the community.

        Returns:
        int: The size of the community.
        """
        return self.size
                        
    def get_members(self):
        """
        Get the members of the community.

        Returns:
        list: The members of the community.
        """
        return self.members
    
    def get_name(self):
        """
        Get the name of the community.

        Returns:
        str: The name of the community.
        """
        return "Community {}".format(self.id)
    
    def get_id(self):
        """
        Get the id of the community.

        Returns:
        int: The id of the community.
        """
        return self.id
    
    def print(self):
        """
        Print the community.
        """
        table = [["Id", "Size", "Sended", "Received", "Volume", "Number of transactions", "Number of unique transactions", "Members",]]
        
        table.append([self.id, self.size, self.sended, self.received, self.volume, self.nb_transactions, self.nb_unique_transactions, self.members])
        print(tabulate.tabulate(table, headers="firstrow"))
        
    def print_transactions(self):
        """
        Print the transactions of this actor.
        """
        table = [["Source", "Target", "Value", "Date", "Nb Transactions"]]
        
        for transaction in self.transactions:
            table.append([transaction.source.name, transaction.target.name, transaction.value, transaction.date, transaction.nb_transactions])
        
        print(tabulate.tabulate(table, headers="firstrow"))