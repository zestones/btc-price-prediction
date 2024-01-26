from utils.transaction import Transaction
from collections import defaultdict
import tabulate

class Actor:
    """
    The Actor class represents an actor in the transaction graph. An actor is a node in the graph that could 
    either has sent or received a transaction. An actor is identified by its name and belongs to a community.
    """
    def __init__(self, name, community):
        """
        Initialize an Actor object.

        Parameters:
        name (str): The name of the actor.
        community (str): The community to which the actor belongs.
        """
        self.name = name
        self.community = community
        
        self.sended = 0
        self.received = 0
        
        self.nb_transactions = 0
        self.nb_unique_transactions = 0
        
        self.transactions = []
        self.transactions_volume_by_day = []

    def process_transaction(self, target, value, nb_transactions, date):
        """
        Process a transaction between this actor and a target actor.

        Parameters:
        target (Actor): The target actor of the transaction.
        value (float): The value of the transaction.
        nb_transactions (int): The number of transactions.
        """
        self.sended += value
        self.nb_transactions += nb_transactions

        target.received += value
        target.nb_unique_transactions += 1

        transaction = Transaction(self, target, value, nb_transactions, date)
        self.transactions.append(transaction)

        return transaction
    
    def process_volume_by_day(self):
        """
        Get the transactions of this actor grouped by date.

        Returns:
        dict: The transactions of this actor grouped by date.
        """
        transactions_by_date = defaultdict(list)  
        for transaction in self.transactions:
            transaction_date = transaction.get_date().date()
            transactions_by_date[transaction_date].append(transaction)
        
        actor = Actor("TOTAL_VOLUME", "-1")
        for transaction_date, transactions in transactions_by_date.items():
            volume_day = sum([transaction.get_value() for transaction in transactions])
            nb_transactions_day = sum([transaction.get_nb_transactions() for transaction in transactions])
            
            transaction = Transaction(self, actor, volume_day, nb_transactions_day, transaction_date)
            self.transactions_volume_by_day.append(transaction)   
         
        self.transactions_volume_by_day = sorted(self.transactions_volume_by_day, key=lambda x: x.get_date())
        return self.transactions_volume_by_day
            
    def set_community(self, community):
        """
        Set the community to which this actor belongs.

        Parameters:
        community (str): The community to which this actor belongs.
        """
        self.community = community
            
    def get_transactions_volume_by_day(self):
        """
        Get the transactions of this actor grouped by date.

        Returns:
        dict: The transactions of this actor grouped by date.
        """
        return self.transactions_volume_by_day
            
    def get_transactions(self):
        """
        Get the transactions of this actor.

        Returns:
        list: The transactions of this actor.
        """
        return self.transactions
        
    def get_total_volume(self):
        """
        Get the total volume of transactions (sent + received) by this actor.

        Returns:
        float: The total volume of transactions.
        """
        return self.received + self.sended
    
    def get_nb_transactions(self):
        """
        Get the nb number of transactions (sent + received) by this actor.

        Returns:
        int: The nb number of transactions.
        """
        return self.nb_transactions
    
    def get_nb_unique_transactions(self):
        """
        Get the nb number of unique transactions (sent + received) by this actor.

        Returns:
        int: The nb number of unique transactions.
        """
        return self.nb_unique_transactions
    
    def get_volume_sended(self):
        """
        Get the total volume of transactions sent by this actor.

        Returns:
        float: The total volume of transactions sent.
        """
        return self.sended
    
    def get_volume_received(self):
        """
        Get the total volume of transactions received by this actor.

        Returns:
        float: The total volume of transactions received.
        """
        return self.received
    
    def get_community(self):
        """
        Get the community to which this actor belongs.

        Returns:
        str: The community to which this actor belongs.
        """
        return self.community
    
    def get_name(self):
        """
        Get the name of this actor.

        Returns:
        str: The name of this actor.
        """
        return self.name
    
    def print(self):
        """
        Print the actor's name and the community to which it belongs.
        """
        table = [["Name", "Community", "Total Volume", "Sent Volume", "Received Volume", "Total Transactions", "Unique Transactions"],
                    [self.name, self.community.get_name(), self.get_total_volume(), self.get_volume_sended(), self.get_volume_received(), self.get_nb_transactions(), self.get_nb_unique_transactions()]]
        
        print(tabulate.tabulate(table, headers="firstrow"))
        
    def print_transactions(self):
        """
        Print the transactions of this actor.
        """
        table = [["Source", "Target", "Value", "Date", "Nb Transactions"]]
        
        for transaction in self.transactions:
            table.append([transaction.source.name, transaction.target.name, transaction.value, transaction.date, transaction.nb_transactions])
        
        print(tabulate.tabulate(table, headers="firstrow"))
