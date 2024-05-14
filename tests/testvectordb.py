import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from VectorStore import VectorDB

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        # Inicializa um novo objeto VectorDB antes de cada teste
        self.vector_db = VectorDB(collection_name="test_collection", data_path="../data")
        self.test_collection_name = "test_collection"
        self.test_document = "This is a test document."
        self.test_embedding = [0.1, 0.2, 0.3]
        self.test_metadata = {"category": "test"}
        self.test_id = "doc1"

        # Criando uma coleção para testes
        self.vector_db.create_collection(self.test_collection_name, embedding_function=None)
        print(f"Setup: Collection '{self.test_collection_name}' created.")

    def test_create_collection(self):
        # Testa se a coleção foi criada corretamente
        print("Testing: Create Collection")
        self.assertIn(self.test_collection_name, self.vector_db.collections)
        print(f"Collection '{self.test_collection_name}' successfully created and verified.")

    def test_get_collection(self):
        # Testa se a coleção pode ser recuperada corretamente
        print("Testing: Get Collection")
        collection = self.vector_db.get_collection(self.test_collection_name)
        self.assertEqual(collection, self.vector_db.collections[self.test_collection_name])
        print(f"Collection '{self.test_collection_name}' successfully retrieved.")

    def test_add_document(self):
        # Testa a adição de um documento
        print("Testing: Add Document")
        self.vector_db.add(self.test_collection_name, [self.test_document], [self.test_embedding], [self.test_metadata], [self.test_id])
        collection = self.vector_db.get_collection(self.test_collection_name)
        self.assertIn(self.test_id, collection["items"])
        self.assertEqual(collection["items"][self.test_id]["document"], self.test_document)
        print(f"Document '{self.test_id}' added with content: {self.test_document}")

    def test_update_document(self):
        # Testa a atualização de um documento
        print("Testing: Update Document")
        new_embedding = [0.4, 0.5, 0.6]
        new_document = "Updated document."
        self.vector_db.add(self.test_collection_name, [self.test_document], [self.test_embedding], [self.test_metadata], [self.test_id])
        self.vector_db.update(self.test_collection_name, [self.test_id], [new_embedding], [self.test_metadata], [new_document])
        collection = self.vector_db.get_collection(self.test_collection_name)
        self.assertEqual(collection["items"][self.test_id]["document"], new_document)
        self.assertEqual(collection["items"][self.test_id]["embedding"], new_embedding)
        print(f"Document '{self.test_id}' updated to new content: {new_document}")

    def test_query(self):
        # Testa a consulta de documentos
        print("Testing: Query Documents")
        self.vector_db.add(self.test_collection_name, [self.test_document], [self.test_embedding], [self.test_metadata], [self.test_id])
        results = self.vector_db.query(self.test_collection_name, [self.test_embedding])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], self.test_id)
        print(f"Query results: {results}")

    def test_delete_document(self):
        # Testa a exclusão de um documento
        print("Testing: Delete Document")
        self.vector_db.add(self.test_collection_name, [self.test_document], [self.test_embedding], [self.test_metadata], [self.test_id])
        self.vector_db.delete(self.test_collection_name, [self.test_id])
        collection = self.vector_db.get_collection(self.test_collection_name)
        self.assertNotIn(self.test_id, collection["items"])
        print(f"Document '{self.test_id}' deleted.")

if __name__ == "__main__":
    unittest.main()
