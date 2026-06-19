import sqlite3

def setup_dummy_db():
    # This creates a file called 'land_investment.db' in your current folder
    conn = sqlite3.connect('land_investment.db')
    cursor = conn.cursor()

    # Create a simple table for land sales
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS land_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            size_perches REAL,
            price_per_perch LKR,
            sale_date TEXT
        )
    ''')

    # Clear old data if you run this multiple times
    cursor.execute('DELETE FROM land_sales')

    # Insert some fake internal data
    sample_data = [
        ('Colombo', 10.5, 4500000, '2026-06-10'),
        ('Colombo', 15.0, 4800000, '2026-06-12'),
        ('Kandy', 20.0, 1500000, '2026-05-20'),
        ('Galle', 12.0, 2200000, '2026-06-01')
    ]
    
    cursor.executemany('''
        INSERT INTO land_sales (city, size_perches, price_per_perch, sale_date)
        VALUES (?, ?, ?, ?)
    ''', sample_data)

    conn.commit()
    conn.close()
    print("✅ Dummy database 'land_investment.db' created successfully!")

if __name__ == "__main__":
    setup_dummy_db()