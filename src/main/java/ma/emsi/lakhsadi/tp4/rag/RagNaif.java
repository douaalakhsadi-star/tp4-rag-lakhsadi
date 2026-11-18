package ma.emsi.lakhsadi.tp4.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class RagNaif {

    public static void main(String[] args) {
        String cle = System.getenv("GEMINI_KEY");

        ChatModel modele = GoogleGenAiChatModel.builder()
                .apiKey(cle)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .build();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println(
                "Caractères extraits du PDF : " + document.text().length()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(1_000, 100);
        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de passages : " + segments.size());

        EmbeddingModel modeleEmbedding = new BgeSmallEnV15EmbeddingModel();

        List<Embedding> embeddings = modeleEmbedding
                .embedAll(segments)
                .content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        EmbeddingStore<TextSegment> magasin = new InMemoryEmbeddingStore<>();
        magasin.addAll(embeddings, segments);

        System.out.println("Passages et embeddings enregistrés en mémoire.");

        ContentRetriever recuperateur = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(magasin)
                .embeddingModel(modeleEmbedding)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ChatMemory memoire = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modele)
                .chatMemory(memoire)
                .contentRetriever(recuperateur)
                .build();

        String question = "Quelle est la signification de 'RAG' ; à quoi ça sert ?";
        String reponse = assistant.chat(question);

        System.out.println("Question : " + question);
        System.out.println("Réponse : " + reponse);
    }
}