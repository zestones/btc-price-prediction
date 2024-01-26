import tabulate

class Transaction:
    """
    This class represents a transaction between two actors. A transaction is an edge in the transaction graph.
    """
    
    def __init__(self, source, target, value, nb_transactions, date):
        """
        Initialize a Transaction object.

        Parameters:
        source (Actor): The source actor of the transaction.
        target (Actor): The target actor of the transaction.
        value (float): The value of the transaction.
        date (datetime): The date of the transaction.
        """
        self.source = source
        self.target = target
        
        self.value = value
        self.date = date
        
        self.nb_transactions = nb_transactions
        
    def get_source(self):
        """
        Get the source actor of the transaction.

        Returns:
        Actor: The source actor.
        """
        return self.source
    
    def get_target(self):
        """
        Get the target actor of the transaction.

        Returns:
        Actor: The target actor.
        """
        return self.target
    
    def get_value(self):
        """
        Get the value of the transaction.

        Returns:
        float: The value of the transaction.
        """
        return self.value
    
    def get_date(self):
        """
        Get the date of the transaction.

        Returns:
        datetime: The date of the transaction.
        """
        return self.date
    
    def get_nb_transactions(self):
        """
        Get the number of transactions.

        Returns:
        int: The number of transactions.
        """
        return self.nb_transactions
    
    def print(self):
        """
        Print the transaction.
        """
        table = [["Source", "Target", "Value", "Date", "Number of transactions"]]
        table.append([self.source.name, self.target.name, self.value, self.date, self.nb_transactions])

        print(tabulate.tabulate(table, headers="firstrow"))