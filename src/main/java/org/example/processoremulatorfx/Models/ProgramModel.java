package org.example.processoremulatorfx.Models;

import org.example.processoremulatorfx.Command;
import org.example.processoremulatorfx.IObserver;

import java.util.*;

public class ProgramModel implements Iterable<Command> {
    ArrayList<Command> allCommands = new ArrayList<>();
    public ArrayList<Command> getAllCommands(){
        return allCommands;
    }
    public int countCommads() { return allCommands.size(); }

    public ProgramModel(){
        allCommands.add(new Command("init", "10", "20"));
        allCommands.add(new Command("init", "11", "25"));
        allCommands.add(new Command("init", "12", "5"));
        allCommands.add(new Command("ld", "a", "10"));
        allCommands.add(new Command("ld", "b", "11"));
        allCommands.add(new Command("ld", "c", "12"));
        allCommands.add(new Command("add"));
        allCommands.add(new Command("print")); // вывод 20 25 5 45
        allCommands.add(new Command("mv", "a", "d"));
        allCommands.add(new Command("mv", "b", "c"));
        allCommands.add(new Command("div"));
        allCommands.add(new Command("print"));  // вывод 45 5 5 9
        eventCall();
    }

    ArrayList<IObserver> allObserver = new ArrayList<>();
    void eventCall(){
        allObserver.forEach(action->action.event(this));
    }
    public void addObserver(IObserver e){
        allObserver.add(e);
        eventCall();
    }

    public void addCommand(Command p){
        allCommands.add(p);
        eventCall();
    }

    public int getNumCommand(Command p){
        return allCommands.indexOf(p);
    }

    public void removeCommand(Command p) {
        allCommands.remove(p);
        eventCall();
    }

    public void switchCommandDown(Command c) throws Exception {
        int indx = allCommands.indexOf(c);
        if (indx != allCommands.size()-1){
            Command tmp = allCommands.get(indx);
            allCommands.set(indx, allCommands.get(indx+1));
            allCommands.set(indx+1, tmp);
            eventCall();
        }
        else throw new Exception("Cannot move down");
    }

    public void switchCommandUp(Command c) throws Exception {
        int indx = allCommands.indexOf(c);
        if (indx != 0){
            Command tmp = allCommands.get(indx);
            allCommands.set(indx, allCommands.get(indx-1));
            allCommands.set(indx-1, tmp);
            eventCall();
        }
        else throw new Exception("Cannot move up");
    }

    public String MostPopularInstruction() {
        Map<String, Integer> commandCount = new HashMap<>();
        // Шаг 2: проходим по массиву команд и считаем их популярность
        for (Command command : allCommands) {
            if (command == null) continue;
            String name = command.getCommand();
            commandCount.put(name, commandCount.getOrDefault(name, 0) + 1);
        }

        // Шаг 3: находим команду с максимальной популярностью
        String mostPopularCommand = null;
        int maxCount = 0;

        for (Map.Entry<String, Integer> entry : commandCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                mostPopularCommand = entry.getKey();
                maxCount = entry.getValue();
            }
        }

        // Шаг 4: создаём список из записей карты и сортируем по значениям (популярности)
        List<Map.Entry<String, Integer>> sortedCommands = new ArrayList<>(commandCount.entrySet());
        sortedCommands.sort((entry1, entry2) -> entry2.getValue().compareTo(entry1.getValue()));

        // Шаг 5: выводим команды в порядке убывания популярности
        int i = 0;
        StringBuilder table = new StringBuilder();
        for (Map.Entry<String, Integer> entry : sortedCommands) {
            String cell = String.format("%" + 5 + "s: " + "%" + 2 + "d\n", entry.getKey(), entry.getValue());
            table.append(cell);
        }
        return table.toString();
    }

    @Override
    public Iterator<Command> iterator() {
        return allCommands.iterator();
    }
}
