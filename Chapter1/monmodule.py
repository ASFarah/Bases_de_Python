"""
Exemple de module python. Contient une variable appelée ma_variable,
Une fonction appelée ma_fonction, et une classe appelée MaClasse.
"""

ma_variable = 0

def ma_fonction():
    """
    Exemple de fonction
    """
    return ma_variable*2
    
class MaClasse:
    """
    Exemple de classe.
    """

    def __init__(self):
        self.variable = ma_variable
        
    def set_variable(self, n_val):
        """
        Définir self.variable à n_val
        """
        self.variable = n_val
        
    def get_variable(self):
        return self.variable
