/**
 * Script to automate login and data extraction from the IDA (Image & Data Archive) web platform using Puppeteer.
 *
 * This script performs the following steps:
 * - Launches a browser and navigates to the IDA homepage.
 * - Accepts the cookie policy popup.
 * - Logs into the platform using provided credentials.
 * - Selects the "ADNI" project.
 * - Navigates to the study files section.
 * - Locates and expands the "Imaging" > "MR Image Analysis" categories.
 * - Searches for specific file names and retrieves their corresponding `data-id` attributes.
 * - Saves the extracted `data-id`s into a CSV file (`data_ids.csv`) for later use.
 *
 * Dependencies:
 * - `puppeteer`: Used to control and automate the browser.
 * - `fs`: For writing the output CSV file.
 *
 * Note:
 * - Set `headless: false` if you want to see the browser interactions for debugging or verification.
 * - Make sure to replace login credentials with secure and environment-based access in production environments.
*/

const puppeteer = require("puppeteer");
const fs = require("fs");

const username = 'username@prova.com';
const password = 'password123';

(async () => {
    // Avvia il browser (cambia headless a false se vuoi vedere il browser in azione)
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();

    // Go to the site
    await page.goto("https://ida.loni.usc.edu/", { waitUntil: "networkidle2" });

    // Force the cookies popup
    await page.evaluate(() => {
        document.querySelector(".ida-cookie-policy-container").classList.add("show");
    });

    // Await cookies
    await page.waitForFunction(() => {
        const parent = document.querySelector(".ida-cookie-policy-container");
        return parent && parent.classList.contains("show");
    }, { timeout: 100000 });

    // Check if the cookies button is visible and click
    await page.waitForSelector(".ida-cookie-policy-container.show .ida-cookie-policy-accept", { visible: true });
    await page.click(".ida-cookie-policy-container.show .ida-cookie-policy-accept");

    await page.waitForNavigation({ waitUntil: "networkidle2" , timeout: 10000}).catch(() => console.log("No reload, but the login should be allowed")); // Wait the reload

    // Click on login
    await page.waitForSelector(".ida-menu-option.sub-menu.login");
    await page.click(".ida-menu-option.sub-menu.login");

    // Fill email and password fields
    await page.waitForSelector('[name="userEmail"]');
    await page.type('[name="userEmail"]', username);

    await page.waitForSelector('[name="userPassword"]');
    await page.type('[name="userPassword"]', password);
    
    // Submit
    await page.keyboard.press("Enter");

    await page.waitForNavigation({ waitUntil: "networkidle2" , timeout: 10000}).catch(() => console.log("No reload, but the project should be allowed")); // Wait the reload

    // Click ADNI project
    await page.waitForSelector('.ida-study-selector-dropdown');
    await page.waitForSelector('.ida-study-selector-study[data-value="ADNI"]');
    await page.evaluate(() => {
        document.querySelector('.ida-study-selector-study[data-value="ADNI"]').click();
    });

    // Go to study files
    await page.goto("https://ida.loni.usc.edu/explore/jsp/search/search.jsp?project=ADNI#studyFiles", { waitUntil: "networkidle2" });
    
    // Wait the table loading
    await page.waitForFunction(() => {
        const elemento = document.querySelector('.ida-animation-container');
        return !elemento || getComputedStyle(elemento).display === 'none' || getComputedStyle(elemento).opacity === '0';
    });    

    // Find images tab
    const category_elem = await page.evaluateHandle(() => {
        return [...document.querySelectorAll(".tree-container .study-category-item")]
            .find(el => el.querySelector(".value")?.innerText.trim() === "Imaging");
    });

    // Click on images tab
    if (category_elem) {
        await category_elem.click(); // Clicca l'elemento trovato
    }

    // Find image analysis subtab
    const sub_category_elem = await page.evaluateHandle(() => {
        return [...document.querySelectorAll(".tree-container .study-category-item.active .study-sub-category-item")]
            .find(el => el.querySelector(".sub-cat-item")?.innerText.trim() === "MR Image Analysis");
    });

    // Click on image analysis subtab
    if (sub_category_elem) {
        await sub_category_elem.click(); // Clicca l'elemento trovato
    }

    // List of files name
    const nomiDaCercare = ["ADNI3 MRI ANALYSIS MANUAL (PDF)", "ASHS volume data [ADNI2,3]"];

    // Get the ids of the files
    const dataIds = await page.evaluate((nomi) => {
        const righe = document.querySelectorAll(".study-item-list tr");
        let risultati = [];
    
        righe.forEach(riga => {
            const nomeTd = riga.querySelector(".item-name");
            if (nomeTd && nomi.includes(nomeTd.innerText.trim())) {
                const dataId = riga.getAttribute("data-id");
                if (dataId) {
                    risultati.push(dataId);
                }
            }
        });
    
        return risultati;
    }, nomiDaCercare);

    console.log(dataIds);

    // Save the ids in a csv
    const csvContent = "data_id\n" + dataIds.join("\n");

    fs.writeFileSync("../data/data_ids.csv", csvContent);

    // Chiude il browser
    await browser.close();
})();
