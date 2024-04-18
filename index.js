import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import readline from 'readline';
import fs from 'fs';
import crypto from 'crypto';
import "dotenv/config";

// Constants
const MODEL_NAME = "gpt-3.5-turbo-1106";
const CSV_FILE_PATH = "docs/sample_qa.csv"; // Path to the CSV file
const HASH_FILE_PATH = "docs/sample_qa_hash.txt"; // Path to the file where the hash is stored
const DIRECTORY = "./vector_db"; // Directory to store the vector store


//###################################################### embeddings ######################################################
const embeddings = new OpenAIEmbeddings();


//###################################################### vectorstore ######################################################
//embed and store in vector store only when there is changes in the csv ###################################################

// Function to calculate the hash of a file
const calculateHash = (filePath) => {
  const fileBuffer = fs.readFileSync(filePath);
  const hashSum = crypto.createHash('sha256');
  hashSum.update(fileBuffer);
  return hashSum.digest('hex');
}

// Calculate the current hash of the CSV file
const currentHash = calculateHash(CSV_FILE_PATH);

let previousHash;

// Check if the hash file exists
if (fs.existsSync(HASH_FILE_PATH)) {
  // If it exists, read the previous hash from the file
  previousHash = fs.readFileSync(HASH_FILE_PATH, 'utf-8');
}


// If the hashes are different or if there was no previous hash, re-index and re-embed the CSV file
if (!previousHash ||currentHash !== previousHash) {

  //CSV file is loaded using the CSVLoader
  const loader = new CSVLoader(CSV_FILE_PATH);
  const docs =  await loader.load();

  const vectorstore_2 = await HNSWLib.fromDocuments(docs, embeddings);
  await vectorstore_2.save(DIRECTORY);

  // Store the current hash in the file
  fs.writeFileSync(HASH_FILE_PATH, currentHash);
}

// Load the vector store from the directory
const loadedVectorStore = await HNSWLib.load(DIRECTORY, embeddings);



//###################################################### retriever ######################################################
const retriever = loadedVectorStore.asRetriever();
//const retriever = vectorstore_1.asRetriever();

//helper function for retrived docs to be string
  const convertDocsToString = (documents) => {
    return documents.map((document) => {
        return `<doc>\n${document.pageContent}\n</doc>`;
    }).join("\n");
};

//document retriever in a chain
const documentRetrievalChain = RunnableSequence.from([
    (input) => input.question,
    retriever,
    convertDocsToString
]);



//###################################################### rephrase question based on history  ######################################################
const REPHRASE_QUESTION_SYSTEM_TEMPLATE = 
  `Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.`;

const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Rephrase the following question as a standalone question:\n{question}"
  ],
]);

const rephraseQuestionChain = RunnableSequence.from([
  rephraseQuestionChainPrompt,
  new ChatOpenAI({ temperature: 0.1, modelName: MODEL_NAME }),
])






//###################################################### Answering questions  ######################################################
const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are an experienced customer service representative, 
expert at interpreting and answering questions based on provided sources.
Using the below provided context and chat history, 
answer the user's question to the best of 
your ability 
using only the resources provided. Be verbose!. If you don't know the answer, just suggest the customer to contact customer service for human assistance.

<context>
{context}
</context>`;

const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
  new MessagesPlaceholder("history"),
  [
    "human", 
    "Now, answer this question using the previous context and chat history:\n{question}"
  ]
]);



//###################################################### Conversational retrieval chain  ######################################################
const conversationalRetrievalChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    standalone_question: rephraseQuestionChain,
  }),
  RunnablePassthrough.assign({
    context: documentRetrievalChain,
  }),
  answerGenerationChainPrompt,
  new ChatOpenAI({ modelName: MODEL_NAME }),
  new StringOutputParser(),
]);

const messageHistory = new ChatMessageHistory();

const finalRetrievalChain = new RunnableWithMessageHistory({
  runnable: conversationalRetrievalChain,
  getMessageHistory: (_sessionId) => messageHistory,
  historyMessagesKey: "history",
  inputMessagesKey: "question",
});


//###################################################### Final Question & Answer Chat Interface  ######################################################

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const askQuestion = () => {
  rl.question('Ask a question: ', async (question) => {
    const finalResult = await finalRetrievalChain.invoke({
      question,
    }, {
      configurable: { sessionId: "test" }
    });

    console.log(`AI Response: ${finalResult}`);
    askQuestion(); // Ask another question
  });
}

askQuestion(); // Start the chat

